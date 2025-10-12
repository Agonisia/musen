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
		const unsigned       iColl              = _collActivityIndices[iActivColl];
		const unsigned       iPart1             = _collSrcIDs[iColl];
		const unsigned       iPart2             = _collDstIDs[iColl];
		const SInteractProps prop               = _interactProps[_collInteractPropIDs[iColl]];
		const CVector3       tangOverlapOld     = _collTangOverlaps[iColl];
		
		// 获取SUP缩放因子
		const double l = m_vConstantModelParameters[0];
		
		// ========== 第一步：从放大颗粒参数转换到原始颗粒参数 ==========
		
		// 1. 几何参数转换
		const double radius1_S = _partRadii[iPart1];      // 放大半径
		const double radius2_S = _partRadii[iPart2];
		const double radius1_O = radius1_S / l;           // 原始半径
		const double radius2_O = radius2_S / l;
		
		// 2. 重叠量转换：δ_O = δ_S / l
		const double overlap_S = _collNormalOverlaps[iColl];
		const double overlap_O = overlap_S / l;
		const double equivRadius_S = _collEquivRadii[iColl];
		const double equivRadius_O = equivRadius_S / l;
		const double equivMass_O = _collEquivMasses[iColl] / (l * l * l);  // 质量缩放 m_O = m_S/l³
		
		// 3. 角速度转换：ω_O = l × ω_S
		const CVector3 anglVel1_S = _partAnglVels[iPart1];
		const CVector3 anglVel2_S = _partAnglVels[iPart2];
		const CVector3 anglVel1_O = anglVel1_S * l;
		const CVector3 anglVel2_O = anglVel2_S * l;
		
		// 4. 接触向量（转换到原始尺度）
		const CVector3 contactVector_S = _collContactVectors[iColl];
		const CVector3 contactVector_O = contactVector_S / l;
		const CVector3 rc1_O = contactVector_O * (radius1_O / (radius1_O + radius2_O));
		const CVector3 rc2_O = contactVector_O * (-radius2_O / (radius1_O + radius2_O));
		const CVector3 normVector = contactVector_S.Normalized();
		
		// 5. 相对速度计算（使用原始参数）
		const CVector3 relVel = (_partVels[iPart2] + anglVel2_O * rc2_O) - 
		                        (_partVels[iPart1] + anglVel1_O * rc1_O);
		const double normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel = normRelVelLen * normVector;
		const CVector3 tangRelVel = relVel - normRelVel;
		
		// ========== 第二步：使用原始参数计算原始颗粒的力和力矩 ==========
		
		// 1. 接触区域半径（基于原始重叠）
		const double contactAreaRadius_O = sqrt(equivRadius_O * overlap_O);
		
		// 2. 法向刚度（原始颗粒）
		const double Kn_O = 2 * prop.dEquivYoungModulus * contactAreaRadius_O;
		
		// 3. 法向力（原始颗粒）
		double normContactForceLen_O;
		if (prop.dEquivSurfaceEnergy > 0) {
			// JKR adhesion model
			const double a3_O = pow(contactAreaRadius_O, 3.0);
			const double elasticForce = 4.0 * a3_O * prop.dEquivYoungModulus / (3.0 * equivRadius_O);
			const double adhesionForce = sqrt(8 * PI * prop.dEquivYoungModulus * 
			                                  prop.dEquivSurfaceEnergy * a3_O);
			normContactForceLen_O = -1.0 * (elasticForce - adhesionForce);
		} else {
			normContactForceLen_O = -2.0 / 3.0 * overlap_O * Kn_O;
		}
		
		const double normDampingForceLen_O = -_2_SQRT_5_6 * prop.dAlpha * normRelVelLen * 
		                                     sqrt(Kn_O * equivMass_O);
		const CVector3 normForce_O = normVector * (normContactForceLen_O + normDampingForceLen_O);
		
		// 4. 切向力（原始颗粒）
		// 旋转旧的切向重叠
		CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
		if (tangOverlapRot.IsSignificant()) {
			tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
		}
		
		// 计算新的切向重叠（使用原始时间步长）
		// 注意：与CPU代码保持一致，暂时不缩放时间步长
		// const double timeStep_O = _timeStep / l;  // 时间步长缩放：Δt_O = Δt_S / l
		const double timeStep_O = _timeStep;  // 暂时不缩放时间步长
		CVector3 tangOverlap_O = tangOverlapRot / l + tangRelVel * timeStep_O;  // 转换到原始尺度
		
		// 切向刚度（原始颗粒）
		const double Kt_O = 8 * prop.dEquivShearModulus * contactAreaRadius_O;
		const CVector3 tangShearForce_O = tangOverlap_O * Kt_O;
		const CVector3 tangDampingForce_O = tangRelVel * (-_2_SQRT_5_6 * prop.dAlpha * 
		                                                   sqrt(Kt_O * equivMass_O));
		
		// 检查滑动条件
		CVector3 tangForce_O;
		const double tangShearForceLen = tangShearForce_O.Length();
		const double frictionForceLen = prop.dSlidingFriction * 
		                                fabs(normContactForceLen_O + normDampingForceLen_O);
		
		if (tangShearForceLen > frictionForceLen) {
			tangForce_O = tangShearForce_O * frictionForceLen / tangShearForceLen;
			tangOverlap_O = tangForce_O / Kt_O;
		} else {
			tangForce_O = tangShearForce_O + tangDampingForce_O;
		}
		
		// 5. 接触力矩（原始颗粒，使用原始半径）
		const CVector3 contactTorque1_O = normVector * tangForce_O * radius1_O;
		const CVector3 contactTorque2_O = normVector * tangForce_O * radius2_O;
		
		// 6. 滚动阻力力矩（原始颗粒，使用原始半径和原始角速度）
		const CVector3 rollingTorque1_O = anglVel1_O.IsSignificant() ? 
			anglVel1_O * (-prop.dRollingFriction * fabs(normContactForceLen_O) * 
			              radius1_O / anglVel1_O.Length()) : CVector3{0};
		const CVector3 rollingTorque2_O = anglVel2_O.IsSignificant() ? 
			anglVel2_O * (-prop.dRollingFriction * fabs(normContactForceLen_O) * 
			              radius2_O / anglVel2_O.Length()) : CVector3{0};
		
		// 7. 总原始力矩
		const CVector3 totalMoment1_O = contactTorque1_O + rollingTorque1_O;
		const CVector3 totalMoment2_O = contactTorque2_O + rollingTorque2_O;
		
		// ========== 第三步：SUP缩放到放大系统 ==========
		
		// 力缩放：F_S = l² × F_O
		const CVector3 totalForce_S = (normForce_O + tangForce_O) * l * l;
		
		// 力矩缩放：M_S = l² × M_O
		const CVector3 moment1_S = totalMoment1_O * l * l;
		const CVector3 moment2_S = totalMoment2_O * l * l;
		
		// ========== 存储结果（注意：切向重叠需要转换回放大尺度）==========
		_collTangOverlaps[iColl] = tangOverlap_O * l;  // 转换回放大尺度存储
		_collTotalForces[iColl] = totalForce_S;
		
		// 应用力和力矩
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart1], totalForce_S);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iPart2], totalForce_S);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart1], moment1_S);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart2], moment2_S);
	}
}