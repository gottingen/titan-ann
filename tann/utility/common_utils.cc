#include "tann/utility/common_utils.h"
#include "tann/distance/DistanceUtils.h"

using namespace tann;
using namespace tann::COMMON;


#define DefineVectorValueType(Name, Type) template int Utils::GetBase<Type>();
#include "tann/utility/DefinitionList.h"
#undef DefineVectorValueType

template <typename T>
void Utils::BatchNormalize(T* data, SizeType row, DimensionType col, int base, int threads) 
{
#pragma omp parallel for num_threads(threads)
	for (SizeType i = 0; i < row; i++)
	{
		tann::COMMON::Utils::Normalize(data + i * (size_t)col, col, base);
	}
}

#define DefineVectorValueType(Name, Type) template void Utils::BatchNormalize<Type>(Type* data, SizeType row, DimensionType col, int base, int threads);
#include "tann/utility/DefinitionList.h"
#undef DefineVectorValueType