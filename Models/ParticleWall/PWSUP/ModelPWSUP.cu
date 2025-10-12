/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWSUP.cuh"
#include "ModelPWSUP.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[1];
__constant__ SPBC PBC;

void CModelPWSUP::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	// 将参数传递到GPU常量内存
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPWSUP::CalculatePWGPU(double _time, double _timeStep, const SInteractProps _interactProps[],
                                  const SGPUParticles& _particles, const SGPUWalls& _walls,
                                  SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_SUP_kernel,
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

__global__ void CUDA_CalcPWForce_SUP_kernel(
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
		const CVector3       normVector      = _wallNormalVecs[iWall];
		const CVector3       tangOverlapOld  = _collTangOverlaps[iColl];
		
		// 获取SUP缩放因子
		const double l = m_vConstantModelParameters[0];
		
		// ========== 第一步：从放大颗粒参数转换到原始颗粒参数 ==========
		
		// 1. 几何参数转换
		const double partRadius_S = _partRadii[iPart];      // 放大半径
		const double partRadius_O = partRadius_S / l;       // 原始半径
		const double partMass_O = _partMasses[iPart] / (l * l * l);  // 质量转换：m_O = m_S/l³
		
		// 2. 角速度转换：ω_O = l × ω_S
		const CVector3 partAnglVel_S = _partAnglVels[iPart];
		const CVector3 partAnglVel_O = partAnglVel_S * l;
		
		// 3. 计算粒子中心到墙接触点的向量（需要考虑缩放）
		const CVector3 particleCoord_S = GPU_GET_VIRTUAL_COORDINATE(_partCoords[iPart]);
		const CVector3 rc_S = particleCoord_S - _collContactPoints[iColl];
		const double rcLen_S = rc_S.Length();
		const CVector3 rcNorm = rc_S / rcLen_S;  // 方向不变
		
		// 转换到原始尺度
		const double rcLen_O = rcLen_S / l;
		
		// 4. 法向重叠量转换：δ_O = δ_S / l
		const double normOverlap_S = partRadius_S - rcLen_S;
		if (normOverlap_S < 0) continue;
		const double normOverlap_O = normOverlap_S / l;
		
		// 5. 相对速度计算（使用原始参数）
		// 墙的速度不变（墙不参与缩放）
		const CVector3 rotVel = !_wallRotVels[iWall].IsZero() ? 
		                        (_collContactPoints[iColl] - _wallRotCenters[iWall]) * _wallRotVels[iWall] : 
		                        CVector3{0};
		
		// 使用原始半径和原始角速度计算相对速度
		const CVector3 relVel = _partVels[iPart] - _wallVels[iWall] + rotVel + 
		                        rcNorm * partAnglVel_O * partRadius_O;  // 使用原始参数
		const double normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel = normRelVelLen * normVector;
		const CVector3 tangRelVel = relVel - normRelVel;
		
		// ========== 第二步：使用原始参数计算原始颗粒的力和力矩 ==========
		
		// 1. 接触区域半径（基于原始重叠和原始半径）
		const double contactAreaRadius_O = sqrt(partRadius_O * normOverlap_O);
		
		// 2. 法向刚度（原始颗粒）
		const double Kn_O = 2 * prop.dEquivYoungModulus * contactAreaRadius_O;
		
		// 3. 法向力（原始颗粒）
		double normContactForceLen_O;
		
		if (prop.dEquivSurfaceEnergy > 0) {
			// JKR adhesion model
			const double a3_O = pow(contactAreaRadius_O, 3.0);
			// PW情况下，等效半径就是粒子的原始半径
			const double elasticForce = 4.0 * a3_O * prop.dEquivYoungModulus / 
			                             (3.0 * partRadius_O);
			const double adhesionForce = sqrt(8 * PI * prop.dEquivYoungModulus * 
			                                  prop.dEquivSurfaceEnergy * a3_O);
			// PW中需要考虑接触方向
			normContactForceLen_O = elasticForce - (adhesionForce * fabs(DotProduct(rcNorm, normVector)));
		} else {
			// 纯Hertz-Mindlin（无粘附）
			normContactForceLen_O = 2.0 / 3.0 * normOverlap_O * Kn_O * fabs(DotProduct(rcNorm, normVector));
		}
		
		// 4. 法向阻尼力（原始颗粒）
		const double normDampingForceLen_O = _2_SQRT_5_6 * prop.dAlpha * normRelVelLen * 
		                                     sqrt(Kn_O * partMass_O);
		const CVector3 normForce_O = normVector * (normContactForceLen_O + normDampingForceLen_O);
		
		// 5. 切向力（原始颗粒）
		// 旋转旧的切向重叠
		CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
		if (tangOverlapRot.IsSignificant()) {
			tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
		}
		
		// 计算新的切向重叠（使用原始时间步长）
		// const double timeStep_O = _timeStep / l;  // 时间步长缩放：Δt_O = Δt_S / l
		const double timeStep_O = _timeStep;  // 暂时不缩放时间步长
		CVector3 tangOverlap_O = tangOverlapRot / l + tangRelVel * timeStep_O;  // 转换到原始尺度
		
		// 切向刚度（原始颗粒）
		const double Kt_O = 8 * prop.dEquivShearModulus * contactAreaRadius_O;
		
		// 注意：PW中切向力符号处理
		const CVector3 tangShearForce_O = -Kt_O * tangOverlap_O;
		const CVector3 tangDampingForce_O = tangRelVel * (_2_SQRT_5_6 * prop.dAlpha * 
		                                                   sqrt(Kt_O * partMass_O));
		
		// 检查滑动条件
		CVector3 tangForce_O;
		const double tangShearForceLen = tangShearForce_O.Length();
		const double frictionForceLen = prop.dSlidingFriction * 
		                                fabs(normContactForceLen_O + normDampingForceLen_O);
		
		if (tangShearForceLen > frictionForceLen) {
			tangForce_O = tangShearForce_O * frictionForceLen / tangShearForceLen;
			tangOverlap_O = tangForce_O / -Kt_O;
		} else {
			tangForce_O = tangShearForce_O + tangDampingForce_O;
		}
		
		// 6. 接触力矩（原始颗粒，使用原始半径）
		const CVector3 contactTorque_O = normVector * tangForce_O * -partRadius_O;
		
		// 7. 滚动阻力力矩（原始颗粒，使用原始半径和原始角速度）
		const CVector3 rollingTorque_O = partAnglVel_O.IsSignificant() ? 
		                                 partAnglVel_O * (-prop.dRollingFriction * 
		                                                  fabs(normContactForceLen_O) * partRadius_O / 
		                                                  partAnglVel_O.Length()) : CVector3{0};
		
		// 8. 总原始力矩
		const CVector3 totalMoment_O = contactTorque_O + rollingTorque_O;
		
		// ========== 第三步：SUP缩放到放大系统 ==========
		
		// 力缩放：F_S = l² × F_O
		const CVector3 totalForce_S = (normForce_O + tangForce_O) * l * l;
		
		// 力矩缩放：M_S = l² × M_O
		const CVector3 moment_S = totalMoment_O * l * l;
		
		// ========== 存储结果（注意：切向重叠需要转换回放大尺度）==========
		_collTangOverlaps[iColl] = tangOverlap_O * l;  // 转换回放大尺度存储
		_collTotalForces[iColl] = totalForce_S;
		
		// 应用力和力矩
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart], moment_S);
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart], totalForce_S);
		CUDA_VECTOR3_ATOMIC_SUB(_wallForces[iWall], totalForce_S);  // 墙只受反作用力
	}
}