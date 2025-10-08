/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWPopovJKR.cuh"
#include "ModelPWPopovJKR.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[1];
__constant__ SPBC PBC;

void CModelPWPopovJKR::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	// 将参数传递到GPU常量内存
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPWPopovJKR::CalculatePWGPU(double _time, double _timeStep, const SInteractProps _interactProps[],
                                  const SGPUParticles& _particles, const SGPUWalls& _walls,
                                  SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_PopovJKR_kernel,
		_timeStep,
		_interactProps,
		
		_particles.AnglVels,
		_particles.Coords,
		_particles.Masses,
		_particles.Radii,
		_particles.Vels,
		_particles.Forces,
		_particles.Moments,
		
		_walls.Vels,
		_walls.RotCenters,
		_walls.RotVels,
		_walls.NormalVectors,
		_walls.Forces,
		
		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.ContactVectors,  // 解释为接触点
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.VirtualShifts,
		
		_collisions.TangOverlaps,
		_collisions.TotalForces
	);
}

__global__ void CUDA_CalcPWForce_PopovJKR_kernel(
	double               _timeStep,
	const SInteractProps _interactProps[],
	
	const CVector3  _partAnglVels[],
	const CVector3  _partCoords[],
	const double    _partMasses[],
	const double    _partRadii[],
	const CVector3  _partVels[],
	CVector3        _partForces[],
	CVector3        _partMoments[],
	
	const CVector3  _wallVels[],
	const CVector3  _wallRotCenters[],
	const CVector3  _wallRotVels[],
	const CVector3  _wallNormalVecs[],
	CVector3        _wallForces[],
	
	const unsigned* _collActiveCollisionsNum,
	const unsigned  _collActivityIndices[],
	const uint16_t  _collInteractPropIDs[],
	const CVector3  _collContactPoints[],
	const unsigned  _collSrcIDs[],
	const unsigned  _collDstIDs[],
	const uint8_t   _collVirtShifts[],
	
	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; 
	     iActivColl < *_collActiveCollisionsNum; 
	     iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned       iColl           = _collActivityIndices[iActivColl];
		const unsigned       iWall           = _collSrcIDs[iColl];
		const unsigned       iPart           = _collDstIDs[iColl];
		const SInteractProps prop            = _interactProps[_collInteractPropIDs[iColl]];
		const double         partRadius      = _partRadii[iPart];
		const CVector3       partAnglVel     = _partAnglVels[iPart];
		const CVector3       normVector      = _wallNormalVecs[iWall];
		const CVector3       tangOverlapOld  = _collTangOverlaps[iColl];
		
		// 获取SUP参数
		const double l     = m_vConstantModelParameters[0];
		
		// 粒子中心到墙接触点的向量
		const CVector3 rc     = GPU_GET_VIRTUAL_COORDINATE(_partCoords[iPart]) - _collContactPoints[iColl];
		const double   rcLen  = rc.Length();
		const CVector3 rcNorm = rc / rcLen;
		
		// 法向重叠
		const double normOverlap = partRadius - rcLen;
		if (normOverlap < 0) continue;
		
		// 墙的旋转速度贡献
		const CVector3 rotVel = !_wallRotVels[iWall].IsZero() ? 
		                        (_collContactPoints[iColl] - _wallRotCenters[iWall]) * _wallRotVels[iWall] : 
		                        CVector3{0};
		
		// 相对速度
		const CVector3 relVel        = _partVels[iPart] - _wallVels[iWall] + rotVel + 
		                                rcNorm * partAnglVel * partRadius;
		const double   normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel    = normRelVelLen * normVector;
		const CVector3 tangRelVel    = relVel - normRelVel;
		
		// 接触区域半径
		const double contactAreaRadius = sqrt(partRadius * normOverlap);
		
		// 修正：SUP模型不缩放材料参数。刚度 Kn 应使用原始杨氏模量。
		const double Kn = 2 * prop.dEquivYoungModulus * contactAreaRadius; // 移除 * l
		
		// 法向力（Hertz-Mindlin + JKR）- 计算原始力 F_NO
		double normContactForceLen;
		if (prop.dEquivSurfaceEnergy > 0) {
			const double a3 = pow(contactAreaRadius, 3.0);
			// 修正：弹性力项不应包含 l 因子
			const double elasticForce = 4.0 * a3 * prop.dEquivYoungModulus / (3.0 * partRadius); // 移除 * l
			// 修正：粘附力项不应包含 l 因子
			const double adhesionForce = sqrt(8 * PI * prop.dEquivYoungModulus * prop.dEquivSurfaceEnergy * a3); // 移除 l*...*l*l
			// PW需要方向项
			normContactForceLen = elasticForce - (adhesionForce * fabs(DotProduct(rcNorm, normVector)));
		} else {
			// 纯Hertz-Mindlin - 使用修正后的 Kn
			normContactForceLen = 2.0 / 3.0 * normOverlap * Kn * fabs(DotProduct(rcNorm, normVector));
		}
		
		const double normDampingForceLen = _2_SQRT_5_6 * prop.dAlpha * normRelVelLen * sqrt(Kn * _partMasses[iPart]); // 使用修正后的 Kn
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);
		
		// 旋转切向重叠
		CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
		if (tangOverlapRot.IsSignificant()) {
			tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
		}
		
		// 新的切向重叠
		CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;
		
		// 修正：SUP模型不缩放材料参数。刚度 Kt 应使用原始剪切模量。
		const double Kt = 8 * prop.dEquivShearModulus * contactAreaRadius; // 移除 * l
		// PW中切向力符号相反
		const CVector3 tangShearForce = -Kt * tangOverlap;
		const CVector3 tangDampingForce = tangRelVel * (_2_SQRT_5_6 * prop.dAlpha * sqrt(Kt * _partMasses[iPart])); // 使用修正后的 Kt
		
		// 滑动检查
		CVector3 tangForce;
		const double tangShearForceLen = tangShearForce.Length();
		const double frictionForceLen = prop.dSlidingFriction * fabs(normContactForceLen + normDampingForceLen);
		
		if (tangShearForceLen > frictionForceLen) {
			tangForce = tangShearForce * frictionForceLen / tangShearForceLen;
			tangOverlap = tangForce / -Kt;  // 注意负号，使用修正后的 Kt
		} else {
			tangForce = tangShearForce + tangDampingForce;
		}
		
		// 滚动阻力 - 计算原始力矩 M_RO
		const CVector3 rollingTorque = partAnglVel.IsSignificant() ? 
		                                partAnglVel * (-prop.dRollingFriction * fabs(normContactForceLen) * partRadius / 
		                                               partAnglVel.Length()) : CVector3{0};
		
		// 应用SUP缩放
		const CVector3 totalForce = (normForce + tangForce) * l * l;  // F_S = l² × F_O (保持不变)
		// 修正：力矩缩放 M_S = l² × M_O
		const CVector3 moment = (normVector * tangForce * -partRadius + rollingTorque) * l * l; // 移除错误的 * l
		
		// 存储结果
		_collTangOverlaps[iColl] = tangOverlap;
		_collTotalForces[iColl]  = totalForce;
		
		// 应用力和力矩
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart], moment);
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart], totalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_wallForces[iWall], totalForce);  // 墙只受反作用力
	}
}