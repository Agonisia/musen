/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWSUP.h"

CModelPWSUP::CModelPWSUP()
{
	std::cout << "Registering PW SUP model..." << std::endl; 
	m_name          = "SUP (Particle-Wall)";
	m_uniqueKey     = "8c9f5e24-9b8d-43af-a12e-518c67295988";  // 需要新的唯一ID
	m_hasGPUSupport = true;
	
	// SUP模型参数
	/* 0*/ AddParameter("SCALE_FACTOR", "SUP scale factor l", 1.0);
}

void CModelPWSUP::CalculatePW(double _time, double _timeStep, size_t _iWall, size_t _iPart,
                               const SInteractProps& _interactProp, SCollision* _collision) const
{
	// 获取SUP缩放因子
	const double l = m_parameters[0].value;
	
	// ========== 第一步：从放大颗粒参数转换到原始颗粒参数 ==========
	
	// 1. 几何参数转换
	const double partRadius_S = Particles().Radius(_iPart);  // 放大半径
	const double partRadius_O = partRadius_S / l;             // 原始半径
	const double partMass_O = Particles().Mass(_iPart) / (l * l * l);  // 质量转换：m_O = m_S/l³
	
	// 2. 角速度转换：ω_O = l × ω_S
	const CVector3 partAnglVel_S = Particles().AnglVel(_iPart);
	const CVector3 partAnglVel_O = partAnglVel_S * l;
	
	// 3. 墙参数（墙不参与缩放）
	const CVector3 normVector = Walls().NormalVector(_iWall);
	
	// 4. 计算粒子中心到墙接触点的向量（需要考虑缩放）
	// 注意：_collision->vContactVector 是放大系统中的接触点位置
	const CVector3 particleCoord_S = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart));
	const CVector3 rc_S = particleCoord_S - _collision->vContactVector;
	const double rcLen_S = rc_S.Length();
	const CVector3 rcNorm = rc_S / rcLen_S;  // 方向不变
	
	// 转换到原始尺度
	const double rcLen_O = rcLen_S / l;
	const CVector3 rc_O = rcNorm * rcLen_O;
	
	// 5. 法向重叠量转换：δ_O = δ_S / l
	const double normOverlap_S = partRadius_S - rcLen_S;
	if (normOverlap_S < 0) return;
	const double normOverlap_O = normOverlap_S / l;
	
	// 6. 相对速度计算（使用原始参数）
	// 墙的速度不变（墙不参与缩放）
	const CVector3 rotVel = !Walls().RotVel(_iWall).IsZero() ? 
	                        (_collision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : 
	                        CVector3{0};
	
	// 使用原始半径和原始角速度计算相对速度
	const CVector3 relVel = Particles().Vel(_iPart) - Walls().Vel(_iWall) + rotVel + 
	                        rcNorm * partAnglVel_O * partRadius_O;  // 使用原始参数
	const double normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel = normRelVelLen * normVector;
	const CVector3 tangRelVel = relVel - normRelVel;
	
	// ========== 第二步：使用原始参数计算原始颗粒的力和力矩 ==========
	
	// 1. 接触区域半径（基于原始重叠和原始半径）
	const double contactAreaRadius_O = std::sqrt(partRadius_O * normOverlap_O);
	
	// 2. 法向刚度（原始颗粒）
	const double Kn_O = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius_O;
	
	// 3. 法向力（原始颗粒）
	double normContactForceLen_O;
	
	if (_interactProp.dEquivSurfaceEnergy > 0) {
		// JKR adhesion model
		const double a3_O = std::pow(contactAreaRadius_O, 3.0);
		// PW情况下，等效半径就是粒子的原始半径
		const double elasticForce = 4.0 * a3_O * _interactProp.dEquivYoungModulus / 
		                             (3.0 * partRadius_O);
		const double adhesionForce = std::sqrt(8 * PI * _interactProp.dEquivYoungModulus * 
		                                       _interactProp.dEquivSurfaceEnergy * a3_O);
		// PW中需要考虑接触方向
		normContactForceLen_O = elasticForce - (adhesionForce * std::abs(DotProduct(rcNorm, normVector)));
	} else {
		// 纯Hertz-Mindlin（无粘附）
		normContactForceLen_O = 2.0 / 3.0 * normOverlap_O * Kn_O * std::abs(DotProduct(rcNorm, normVector));
	}
	
	// 4. 法向阻尼力（原始颗粒）
	const double normDampingForceLen_O = _2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * 
	                                     std::sqrt(Kn_O * partMass_O);
	const CVector3 normForce_O = normVector * (normContactForceLen_O + normDampingForceLen_O);
	
	// 5. 切向力（原始颗粒）
	// 旋转旧的切向重叠
	CVector3 tangOverlapRot = _collision->vTangOverlap - 
	                          normVector * DotProduct(normVector, _collision->vTangOverlap);
	if (tangOverlapRot.IsSignificant())
		tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
	
	// 计算新的切向重叠（使用原始时间步长）
	// const double timeStep_O = _timeStep / l;  // 时间步长缩放：Δt_O = Δt_S / l
	const double timeStep_O = _timeStep;  // 暂时不缩放时间步长
	CVector3 tangOverlap_O = tangOverlapRot / l + tangRelVel * timeStep_O;  // 转换到原始尺度
	
	// 切向刚度（原始颗粒）
	const double Kt_O = 8 * _interactProp.dEquivShearModulus * contactAreaRadius_O;
	
	// 注意：PW中切向力符号处理
	const CVector3 tangShearForce_O = -Kt_O * tangOverlap_O;
	const CVector3 tangDampingForce_O = tangRelVel * (_2_SQRT_5_6 * _interactProp.dAlpha * 
	                                                   std::sqrt(Kt_O * partMass_O));
	
	// 检查滑动条件
	CVector3 tangForce_O;
	const double tangShearForceLen = tangShearForce_O.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * 
	                                std::abs(normContactForceLen_O + normDampingForceLen_O);
	
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
	                                 partAnglVel_O * (-_interactProp.dRollingFriction * 
	                                                  std::abs(normContactForceLen_O) * partRadius_O / 
	                                                  partAnglVel_O.Length()) : CVector3{0};
	
	// 8. 总原始力矩
	const CVector3 totalMoment_O = contactTorque_O + rollingTorque_O;
	
	// ========== 第三步：SUP缩放到放大系统 ==========
	
	// 力缩放：F_S = l² × F_O
	const CVector3 totalForce_S = (normForce_O + tangForce_O) * l * l;
	
	// 力矩缩放：M_S = l² × M_O
	const CVector3 moment_S = totalMoment_O * l * l;
	
	// ========== 存储结果（注意：切向重叠需要转换回放大尺度）==========
	_collision->vTangOverlap = tangOverlap_O * l;  // 转换回放大尺度存储
	_collision->vTangForce = tangForce_O * l * l;  // 存储缩放后的切向力
	_collision->vTotalForce = totalForce_S;
	_collision->vResultMoment1 = moment_S;  // 只有粒子受力矩
}

void CModelPWSUP::ConsolidatePart(double _time, double _timeStep, size_t _iPart,
                                   SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart)  += _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment1;
}

void CModelPWSUP::ConsolidateWall(double _time, double _timeStep, size_t _iWall,
                                   SWallStruct& _walls, const SCollision* _collision) const
{
	// 墙只受反作用力，不受力矩
	_walls.Force(_iWall) -= _collision->vTotalForce;
}