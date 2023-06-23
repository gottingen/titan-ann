//
// Copyright (C) 2020 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "tann/ngt/ngtq/quantized_graph.h"
#include "tann/ngt/ngtq/quantized_blob_graph.h"
#include "tann/ngt/ngtq/optimizer.h"

#ifdef NGTQ_QBG

void
NGTQG::Index::quantize(const std::string indexPath, size_t dimensionOfSubvector, size_t maxNumOfEdges, bool silence) {
    {
        tann::Index index(indexPath);
        const std::string quantizedIndexPath = indexPath + "/qg";
        struct stat st;
        if (stat(quantizedIndexPath.c_str(), &st) != 0) {
            tann::Property ngtProperty;
            index.getProperty(ngtProperty);
            QBG::BuildParameters buildParameters;
            buildParameters.creation.dimensionOfSubvector = dimensionOfSubvector;
            buildParameters.silence = silence;

            NGTQG::Index::create(indexPath, buildParameters);

            NGTQG::Index::append(indexPath, buildParameters);

            QBG::Optimizer optimizer(buildParameters);
#ifdef NGTQG_NO_ROTATION
            if (optimizer.rotation || optimizer.repositioning) {
          std::cerr << "build-qg: Warning! Although rotation or repositioning is specified, turn off rotation and repositioning because of unavailable options." << std::endl;
          optimizer.rotation = false;
          optimizer.repositioning = false;
            }
#endif

            if (optimizer.globalType == QBG::Optimizer::GlobalTypeNone) {
                std::cerr
                        << "build-qg: Warning! None is unavailable for the global type. Zero is set to the global type."
                        << std::endl;
                optimizer.globalType = QBG::Optimizer::GlobalTypeZero;
            }

            optimizer.optimize(quantizedIndexPath);

            QBG::Index::buildNGTQ(quantizedIndexPath, silence);

            NGTQG::Index::realign(indexPath, maxNumOfEdges, silence);
        }
    }

}

void NGTQG::Index::create(const std::string indexPath, QBG::BuildParameters &buildParameters) {
    auto dimensionOfSubvector = buildParameters.creation.dimensionOfSubvector;
    auto dimension = buildParameters.creation.dimension;
    if (dimension != 0 && buildParameters.creation.localDivisionNo != 0) {
        if (dimension % buildParameters.creation.localDivisionNo != 0) {
            std::stringstream msg;
            msg << "NGTQBG:Index::create: Invalid dimension and local division No. " << dimension << ":"
                << buildParameters.creation.localDivisionNo;
            NGTThrowException(msg);
        }
        dimensionOfSubvector = dimension / buildParameters.creation.localDivisionNo;
    }
    create(indexPath, dimensionOfSubvector, dimension);
}


void NGTQG::Index::append(const std::string indexPath, QBG::BuildParameters &buildParameters) {
    QBG::Index::appendFromObjectRepository(indexPath, indexPath + "/qg", buildParameters.silence);
}

#endif
