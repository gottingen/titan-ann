#include <tann/quantizer/i_quantizer.h>
#include <tann/quantizer/pq_quantizer.h>
#include <tann/quantizer/opq_quantizer.h>
#include <tann/utility/StringConvert.h>
#include "turbo/format/print.h"

namespace tann {
    namespace COMMON {
        std::shared_ptr<IQuantizer> IQuantizer::LoadIQuantizer(std::shared_ptr<Helper::DiskIO> p_in) {
            QuantizerType quantizerType = QuantizerType::Undefined;
            VectorValueType reconstructType = VectorValueType::Undefined;
            std::shared_ptr<IQuantizer> ret = nullptr;
            if (p_in->ReadBinary(sizeof(QuantizerType), (char *) &quantizerType) != sizeof(QuantizerType)) return ret;
            if (p_in->ReadBinary(sizeof(VectorValueType), (char *) &reconstructType) != sizeof(VectorValueType))
                return ret;
            TLOG_INFO("Loading quantizer of type %s with reconstructtype {}.",
                         Helper::Convert::ConvertToString<QuantizerType>(quantizerType).c_str(),
                         Helper::Convert::ConvertToString<VectorValueType>(reconstructType).c_str());
            switch (quantizerType) {
                case QuantizerType::None:
                    break;
                case QuantizerType::Undefined:
                    break;
                case QuantizerType::PQQuantizer:
                    turbo::Println("Resetting Quantizer to type PQQuantizer!");
                    switch (reconstructType) {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        ret.reset(new PQQuantizer<Type>()); \
                        break;

#include "tann/utility/DefinitionList.h"

#undef DefineVectorValueType

                        default:
                            break;
                    }

                    if (ret->LoadQuantizer(p_in) != ErrorCode::Success) ret.reset();
                    return ret;
                case QuantizerType::OPQQuantizer:
                    switch (reconstructType) {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        ret.reset(new OPQQuantizer<Type>()); \
                        break;

#include "tann/utility/DefinitionList.h"

#undef DefineVectorValueType
                        default:
                            break;
                    }
                    if (ret->LoadQuantizer(p_in) != ErrorCode::Success) ret.reset();
                    return ret;
            }
            return ret;
        }

        std::shared_ptr<IQuantizer> IQuantizer::LoadIQuantizer(tann::ByteArray bytes) {
            auto raw_bytes = bytes.Data();
            QuantizerType quantizerType = *(QuantizerType *) raw_bytes;
            raw_bytes += sizeof(QuantizerType);

            VectorValueType reconstructType = *(VectorValueType *) raw_bytes;
            raw_bytes += sizeof(VectorValueType);
            TLOG_INFO("Loading quantizer of type {} with reconstructtype {}.",
                         Helper::Convert::ConvertToString<QuantizerType>(quantizerType).c_str(),
                         Helper::Convert::ConvertToString<VectorValueType>(reconstructType).c_str());
            std::shared_ptr<IQuantizer> ret = nullptr;

            switch (quantizerType) {
                case QuantizerType::None:
                    return ret;
                case QuantizerType::Undefined:
                    return ret;
                case QuantizerType::PQQuantizer:
                    switch (reconstructType) {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        ret.reset(new PQQuantizer<Type>()); \
                        break;

#include "tann/utility/DefinitionList.h"

#undef DefineVectorValueType
                        default:
                            break;
                    }

                    if (ret->LoadQuantizer(raw_bytes) != ErrorCode::Success) ret.reset();
                    return ret;
                case QuantizerType::OPQQuantizer:
                    switch (reconstructType) {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        ret.reset(new OPQQuantizer<Type>()); \
                        break;

#include "tann/utility/DefinitionList.h"

#undef DefineVectorValueType
                        default:
                            break;
                    }

                    if (ret->LoadQuantizer(raw_bytes) != ErrorCode::Success) ret.reset();
                    return ret;
            }
            return ret;
        }

        template<>
        std::function<float(const std::uint8_t *, const std::uint8_t *, SizeType)>
        IQuantizer::DistanceCalcSelector<std::uint8_t>(tann::DistCalcMethod p_method) const {
            if (p_method == tann::DistCalcMethod::L2) {
                return ([this](const std::uint8_t *pX, const std::uint8_t *pY,
                               DimensionType length) { return L2Distance(pX, pY); });
            } else {
                return ([this](const std::uint8_t *pX, const std::uint8_t *pY,
                               DimensionType length) { return CosineDistance(pX, pY); });
            }
        }

        template<typename T>
        std::function<float(const T *, const T *, SizeType)>
        IQuantizer::DistanceCalcSelector(tann::DistCalcMethod p_method) const {
            return tann::COMMON::DistanceCalcSelector<T>(p_method);
        }

#define DefineVectorValueType(Name, Type) \
template std::function<float(const Type*, const Type*, SizeType)> IQuantizer::DistanceCalcSelector<Type>(tann::DistCalcMethod p_method) const;

#include "tann/utility/DefinitionList.h"

#undef DefineVectorValueType
    }
}
