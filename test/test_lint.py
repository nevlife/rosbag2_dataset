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
