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
	/* 1*/ AddParameter("SURFACE_ENERGY", "Surface energy density [J/m²]", 0.0);
}

void CModelPWSUP::CalculatePW(double _time, double _timeStep, size_t _iWall, size_t _iPart,
                               const SInteractProps& _interactProp, SCollision* _collision) const
{
	// 获取SUP参数
	const double l = m_parameters[0].value;      // SUP缩放因子
	const double gamma = m_parameters[1].value;  // 表面能密度 (JKR粘附)
	
	// 获取粒子属性
	const double   partRadius  = Particles().Radius(_iPart);
	const CVector3 partAnglVel = Particles().AnglVel(_iPart);
	const CVector3 normVector  = Walls().NormalVector(_iWall);
	
	// 计算粒子中心到墙接触点的向量
	const CVector3 rc     = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _collision->vContactVector;
	const double   rcLen  = rc.Length();
	const CVector3 rcNorm = rc / rcLen;
	
	// 法向重叠
	const double normOverlap = partRadius - rcLen;
	if (normOverlap < 0) return;
	
	// 考虑墙的运动（包括旋转）
	const CVector3 rotVel = !Walls().RotVel(_iWall).IsZero() ? 
	                        (_collision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : 
	                        CVector3{ 0 };
	
	// 法向和切向相对速度
	const CVector3 relVel        = Particles().Vel(_iPart) - Walls().Vel(_iWall) + rotVel + 
	                                rcNorm * partAnglVel * partRadius;
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;
	
	// 接触区域半径（基于实际重叠）
	const double contactAreaRadius = std::sqrt(partRadius * normOverlap);
	
	// SUP缩放的法向刚度
	const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius * l;
	
	// 计算法向力（Hertz-Mindlin + JKR粘附）
	double normContactForceLen;
	
	if (gamma > 0) {
		// JKR粘附力（SUP缩放）
		const double a3 = std::pow(contactAreaRadius, 3.0);
		// 注意：对于PW，等效半径应该是粒子半径（墙视为无限半径）
		const double elasticForce = 4.0 * a3 * _interactProp.dEquivYoungModulus * l / 
		                             (3.0 * partRadius);
		const double adhesionForce = std::sqrt(8 * PI * _interactProp.dEquivYoungModulus * l * 
		                                        gamma * l * l * a3);  // γ缩放为l²
		// PW中需要加上方向项
		normContactForceLen = (elasticForce - adhesionForce) * std::abs(DotProduct(rcNorm, normVector));
	} else {
		// 纯Hertz-Mindlin（无粘附）- 注意PW中的符号和方向处理
		normContactForceLen = 2.0 / 3.0 * normOverlap * Kn * std::abs(DotProduct(rcNorm, normVector));
	}
	
	// 法向阻尼力
	const double normDampingForceLen = _2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * 
	                                    std::sqrt(Kn * Particles().Mass(_iPart));
	const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);
	
	// 旋转旧的切向重叠
	CVector3 tangOverlapRot = _collision->vTangOverlap - 
	                          normVector * DotProduct(normVector, _collision->vTangOverlap);
	if (tangOverlapRot.IsSignificant())
		tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
	
	// 计算新的切向重叠
	CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;
	
	// SUP缩放的切向刚度
	const double Kt = 8 * _interactProp.dEquivShearModulus * contactAreaRadius * l;
	// 注意：PW中切向力符号相反
	const CVector3 tangShearForce = -Kt * tangOverlap;
	const CVector3 tangDampingForce = tangRelVel * 
	                                   (_2_SQRT_5_6 * _interactProp.dAlpha * 
	                                    std::sqrt(Kt * Particles().Mass(_iPart)));
	
	// 检查滑动条件
	CVector3 tangForce;
	const double tangShearForceLen = tangShearForce.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * 
	                                 std::abs(normContactForceLen + normDampingForceLen);
	
	if (tangShearForceLen > frictionForceLen) {
		tangForce = tangShearForce * frictionForceLen / tangShearForceLen;
		tangOverlap = tangForce / -Kt;  // 注意负号
	} else {
		tangForce = tangShearForce + tangDampingForce;
	}
	
	// 滚动阻力（SUP缩放）- 只作用于粒子
	const CVector3 rollingTorque = partAnglVel.IsSignificant() ? 
	                                partAnglVel * (-_interactProp.dRollingFriction * 
	                                               std::abs(normContactForceLen) * partRadius / 
	                                               partAnglVel.Length()) : CVector3{0};
	
	// 应用SUP缩放到最终的力和力矩
	const CVector3 totalForce = (normForce + tangForce) * l * l;  // 力缩放 l²
	// PW中力矩计算：注意符号和缩放
	const CVector3 moment = (normVector * tangForce * -partRadius + rollingTorque) * l * l * l;  // 力矩缩放 l³
	
	// 存储结果
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce * l * l;  // 存储缩放后的切向力
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment;  // 只有粒子受力矩
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