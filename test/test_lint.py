# Copyright 2026 nevlife
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Apply the repository's ament Python style checks."""

from pathlib import Path

import pytest


@pytest.mark.linter
@pytest.mark.flake8
def test_flake8():
    from ament_flake8.main import main_with_errors

    code, errors = main_with_errors(argv=[str(Path(__file__).parents[1])])
    assert code == 0, '\n'.join(errors)


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    from ament_pep257.main import main

    assert main(argv=[str(Path(__file__).parents[1])]) == 0


@pytest.mark.linter
@pytest.mark.copyright
def test_copyright():
    from ament_copyright.main import main

    assert main(argv=[str(Path(__file__).parents[1])]) == 0
