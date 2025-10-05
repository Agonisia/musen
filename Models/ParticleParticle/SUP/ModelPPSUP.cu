/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSUP.cuh"
#include "ModelPPSUP.h"
#include <device_launch_parameters.h>

// 只需要一个参数：SCALE_FACTOR
__constant__ double m_vConstantModelParameters[1];
__constant__ SPBC PBC;

void CModelPPSUP::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	// 将缩放因子传递到GPU常量内存
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPPSUP::CalculatePPGPU(double _time, double _timeStep, const SInteractProps _interactProps[], 
                                  const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_SUP_kernel,
		_timeStep,
		_interactProps,
		
		_particles.AnglVels,
		_particles.Radii,
		_particles.Vels,
		_particles.Forces,
		_particles.Moments,
		
		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.EquivMasses,
		_collisions.EquivRadii,
		_collisions.NormalOverlaps,
		_collisions.ContactVectors,
		
		_collisions.TangOverlaps,
		_collisions.TotalForces
	);
}

__global__ void CUDA_CalcPPForce_SUP_kernel(
	double                  _timeStep,
	const SInteractProps    _interactProps[],
	
	const CVector3  _partAnglVels[],
	const double    _partRadii[],
	const CVector3  _partVels[],
	CVector3        _partForces[],
	CVector3        _partMoments[],
	
	const unsigned* _collActiveCollisionsNum,
	const unsigned  _collActivityIndices[],
	const uint16_t  _collInteractPropIDs[],
	const unsigned  _collSrcIDs[],
	const unsigned  _collDstIDs[],
	const double    _collEquivMasses[],
	const double    _collEquivRadii[],
	const double    _collNormalOverlaps[],
	const CVector3  _collContactVectors[],
	
	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; 
		iActivColl < *_collActiveCollisionsNum; 
		iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned       iColl         = _collActivityIndices[iActivColl];
		const unsigned       iPart1        = _collSrcIDs[iColl];
		const unsigned       iPart2        = _collDstIDs[iColl];
		const SInteractProps prop          = _interactProps[_collInteractPropIDs[iColl]];
		const double         normOverlap   = _collNormalOverlaps[iColl];
		const double         equivMass     = _collEquivMasses[iColl];
		const double         equivRadius   = _collEquivRadii[iColl];
		const CVector3       anglVel1      = _partAnglVels[iPart1];
		const CVector3       anglVel2      = _partAnglVels[iPart2];
		const double         radius1       = _partRadii[iPart1];
		const double         radius2       = _partRadii[iPart2];
		const CVector3       contactVector = _collContactVectors[iColl];
		const CVector3       tangOverlapOld = _collTangOverlaps[iColl];
		
		// 获取SUP缩放因子（从常量内存）
		const double l = m_vConstantModelParameters[0];
		
		const CVector3 rc1        = contactVector * (radius1 / (radius1 + radius2));
		const CVector3 rc2        = contactVector * (-radius2 / (radius1 + radius2));
		const CVector3 normVector = contactVector.Normalized();
		
		// 相对速度
		const CVector3 relVel        = (_partVels[iPart2] + anglVel2 * rc2) - 
		                                (_partVels[iPart1] + anglVel1 * rc1);
		const double   normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel    = normRelVelLen * normVector;
		const CVector3 tangRelVel    = relVel - normRelVel;
		
		// 接触区域半径
		const double contactAreaRadius = sqrt(equivRadius * normOverlap);
		
		// 修正：SUP模型不缩放材料参数。刚度 Kn 应使用原始杨氏模量。
		const double Kn = 2 * prop.dEquivYoungModulus * contactAreaRadius;
		
		// 法向力（Hertz-Mindlin + JKR）- 计算原始力 F_NO
		double normContactForceLen;
		if (prop.dEquivSurfaceEnergy > 0) {
			const double a3 = pow(contactAreaRadius, 3.0);
			// 修正：弹性力项不应包含 l 因子
			const double elasticForce = 4.0 * a3 * prop.dEquivYoungModulus / (3.0 * equivRadius);
			// 修正：粘附力项不应包含 l 因子
			const double adhesionForce = sqrt(8 * PI * prop.dEquivYoungModulus * prop.dEquivSurfaceEnergy * a3);
			normContactForceLen = -1.0 * (elasticForce - adhesionForce);
		} else {
			// 使用修正后的 Kn
			normContactForceLen = 2.0 / 3.0 * normOverlap * Kn;
		}
		
		const double normDampingForceLen = -_2_SQRT_5_6 * prop.dAlpha * normRelVelLen * sqrt(Kn * equivMass);
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);
		
		// 旋转切向重叠
		CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
		if (tangOverlapRot.IsSignificant()) {
			tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
		}

		// 新的切向重叠
		CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;
		
		// 修正：SUP模型不缩放材料参数。刚度 Kt 应使用原始剪切模量。
		const double Kt = 8 * prop.dEquivShearModulus * contactAreaRadius;
		const CVector3 tangShearForce = tangOverlap * Kt;
		const CVector3 tangDampingForce = tangRelVel * (-_2_SQRT_5_6 * prop.dAlpha * sqrt(Kt * equivMass));
		
		// 滑动检查
		CVector3 tangForce;
		const double tangShearForceLen = tangShearForce.Length();
		const double frictionForceLen = prop.dSlidingFriction * fabs(normContactForceLen + normDampingForceLen);
		
		if (tangShearForceLen > frictionForceLen) {
			tangForce = tangShearForce * frictionForceLen / tangShearForceLen;
			tangOverlap = tangForce / Kt; // 使用修正后的 Kt
		} else {
			tangForce = tangShearForce + tangDampingForce;
		}
		
		// 滚动阻力 - 计算原始力矩 M_RO
		const CVector3 rollingTorque1 = anglVel1.IsSignificant() ? 
		        anglVel1 * (-prop.dRollingFriction * fabs(normContactForceLen) * radius1 / anglVel1.Length()) : CVector3{0};
		const CVector3 rollingTorque2 = anglVel2.IsSignificant() ? 
		        anglVel2 * (-prop.dRollingFriction * fabs(normContactForceLen) * radius2 / anglVel2.Length()) : CVector3{0};
		
		// 应用SUP缩放
		// 1. 力缩放：F_S = l² × F_O （保持不变）
		const CVector3 totalForce    = (normForce + tangForce) * l * l;
        
		// 2. 力矩缩放：M_S = l² × M_O （修正为 l²）
		const CVector3 resultMoment1 = (normVector * tangForce * radius1 + rollingTorque1) * l * l; // 移除错误的 * l
		const CVector3 resultMoment2 = (normVector * tangForce * radius2 + rollingTorque2) * l * l; // 移除错误的 * l
		
		// 存储结果
		_collTangOverlaps[iColl] = tangOverlap;
		_collTotalForces[iColl]  = totalForce;
		
		// 应用力和力矩
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart1], totalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iPart2], totalForce);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart1], resultMoment1);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart2], resultMoment2);
	}
}