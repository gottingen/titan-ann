// Copyright 2023 The titan-search Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "tann/core/vector_set_io.h"

namespace tann {

    turbo::Status VectorSetReader::initialize(turbo::SequentialReadFile *file, const SerializeOption &option) {
        _file = file;
        _option = option;
        _element_size = data_type_size(_option.data_type);
        _vector_bytes = option.dimension * _element_size;
        return init();
    }

    turbo::Status VectorSetWriter::initialize(turbo::SequentialWriteFile *file, const SerializeOption &option) {
        _file = file;
        _option = option;
        _element_size = data_type_size(_option.data_type);
        _vector_bytes = _option.dimension * _element_size;
        return init();
    }
}  // namespace tann
