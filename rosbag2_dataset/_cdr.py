# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile ros2msg definitions into XCDR1 buffer decoders."""

from functools import lru_cache
import keyword
import re
import struct

import numpy as np


_PRIMITIVES = {
    'bool': ('?', '?'), 'byte': ('B', 'u1'), 'char': ('B', 'u1'),
    'int8': ('b', 'i1'), 'uint8': ('B', 'u1'),
    'int16': ('h', 'i2'), 'uint16': ('H', 'u2'),
    'int32': ('i', 'i4'), 'uint32': ('I', 'u4'),
    'int64': ('q', 'i8'), 'uint64': ('Q', 'u8'),
    'float32': ('f', 'f4'), 'float64': ('d', 'f8'),
}
_FIELD = re.compile(r'^(\S+)\s+([a-zA-Z_]\w*)(.*)$')
_TYPE = re.compile(r'^(\w+(?:/\w+)*)(?:<=\d+)?(?:\[(\d*|<=\d+)\])?$')


class Message:
    """Hold decoded fields; numeric arrays retain their original buffer."""

    __slots__ = ()
    _fields = ()

    def to_dict(self):
        """Return plain dictionaries and lists, copying numeric arrays."""
        def convert(value):
            if isinstance(value, Message):
                return value.to_dict()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return {name: convert(getattr(self, name)) for name in self._fields}


def _canonical(name, package=''):
    parts = name.split('/')
    if len(parts) == 1:
        return f'{package}/msg/{name}'
    return f'{parts[0]}/msg/{parts[-1]}'


def _definitions(name, text):
    sections = re.split(r'^={10,}\s*\nMSG:\s*(\S+)\s*\n', text, flags=re.M)
    definitions = {_canonical(name): sections[0]}
    definitions.update({_canonical(sections[i]): sections[i + 1]
                        for i in range(1, len(sections), 2)})
    parsed = {}
    for typename, definition in definitions.items():
        fields = []
        for line in definition.splitlines():
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            match = _FIELD.fullmatch(line)
            if match is None:
                raise ValueError(f'Invalid ros2msg field: {line!r}')
            token, field, rest = match.groups()
            if rest.lstrip().startswith('='):
                continue
            if keyword.iskeyword(field) or field.startswith('_') or field == 'to_dict':
                raise ValueError(f'Unsupported field name: {field!r}')
            match = _TYPE.fullmatch(token)
            if match is None:
                raise ValueError(f'Unsupported ros2msg type: {token!r}')
            base, array = match.groups()
            if base not in _PRIMITIVES and base not in ('string', 'wstring'):
                base = _canonical(base, typename.split('/')[0])
            fields.append((field, base, array))
        parsed[typename] = fields
    return parsed


def _wstring(buffer, offset, length, endian):
    units = np.frombuffer(buffer, dtype=endian + 'u4', count=length, offset=offset)
    return units.astype('<u2').tobytes().decode('utf-16-le')


@lru_cache(maxsize=128)
def compile_decoder(name, text):
    """Compile both byte orders once for a schema, without importing ROS."""
    definitions = _definitions(name, text)
    ids = {typename: i for i, typename in enumerate(definitions)}
    decoders = {}
    for endian in ('<', '>'):
        namespace = {'_np': np, '_wstring': _wstring}
        lines = []
        for typename, fields in definitions.items():
            ident = ids[typename]
            namespace[f'_cls{ident}'] = type(
                typename.split('/')[-1], (Message,),
                {'__slots__': tuple(f[0] for f in fields),
                 '_fields': tuple(f[0] for f in fields)})
            lines.extend([f'def _d{ident}(b, p):',
                          f'    m = _cls{ident}.__new__(_cls{ident})'])

            def emit(code, indent=1):
                lines.append('    ' * indent + code)

            def scalar(base, target, indent):
                if base in _PRIMITIVES:
                    fmt, dtype = _PRIMITIVES[base]
                    size = np.dtype(dtype).itemsize
                    key = '_unpack_' + base
                    namespace[key] = struct.Struct(endian + fmt).unpack_from
                    emit(f'p = (p + {size - 1}) & -{size}', indent)
                    if base == 'byte':
                        emit(f'{target} = bytes(b[p:p + 1])', indent)
                    else:
                        emit(f'{target} = {key}(b, p)[0]', indent)
                    emit(f'p += {size}', indent)
                elif base in ('string', 'wstring'):
                    namespace['_u32'] = struct.Struct(endian + 'I').unpack_from
                    emit('p = (p + 3) & -4', indent)
                    emit('length = _u32(b, p)[0]', indent)
                    emit('p += 4', indent)
                    if base == 'string':
                        emit(f'{target} = bytes(b[p:p + length - 1]).decode("utf-8")',
                             indent)
                        emit('p += length', indent)
                    else:
                        emit(f'{target} = _wstring(b, p, length, {endian!r})', indent)
                        emit('p += length * 4', indent)
                else:
                    if base not in ids:
                        raise ValueError(f'Missing dependent ros2msg definition: {base}')
                    emit(f'{target}, p = _d{ids[base]}(b, p)', indent)

            for field, base, array in fields:
                target = f'm.{field}'
                if array is None:
                    scalar(base, target, 1)
                    continue
                if array == '' or array.startswith('<='):
                    namespace['_u32'] = struct.Struct(endian + 'I').unpack_from
                    emit('p = (p + 3) & -4')
                    emit('n = _u32(b, p)[0]')
                    emit('p += 4')
                else:
                    emit(f'n = {int(array)}')
                if base in _PRIMITIVES:
                    dtype = np.dtype(endian + _PRIMITIVES[base][1])
                    size = dtype.itemsize
                    key = '_dtype_' + base
                    namespace[key] = dtype
                    # Empty sequences do not introduce element alignment in CDR.
                    emit('if n:')
                    emit(f'p = (p + {size - 1}) & -{size}', 2)
                    emit(f'{target} = _np.frombuffer(b, {key}, n, p)')
                    emit(f'{target}.flags.writeable = False')
                    emit(f'p += n * {size}')
                else:
                    emit(f'{target} = []')
                    emit('for _ in range(n):')
                    scalar(base, 'value', 2)
                    emit(f'{target}.append(value)', 2)
            if not fields:
                emit('p += 1')
            emit('return m, p')
        exec(compile('\n'.join(lines), '<ros2msg decoder>', 'exec'), namespace)
        decoders[endian] = namespace[f'_d{ids[_canonical(name)]}']

    def decode(payload):
        if len(payload) < 4 or bytes(payload[:2]) not in (b'\x00\x00', b'\x00\x01'):
            raise ValueError('Expected XCDR1 plain CDR encapsulation (CDR_BE or CDR_LE)')
        buffer = memoryview(payload).toreadonly()[4:]
        return decoders['<' if payload[1] == 1 else '>'](buffer, 0)[0]

    return decode
