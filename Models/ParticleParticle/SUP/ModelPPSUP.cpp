/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSUP.h"

CModelPPSUP::CModelPPSUP()
{
	std::cout << "Registering SUP model..." << std::endl; 
	m_name          = "SUP";
	m_uniqueKey     = "77b04f23-8a7b-42ae-9ee7-417b56184877";  // 需要生成唯一ID
	m_hasGPUSupport = true;
    
	// SUP模型参数
	/* 0*/ AddParameter("SCALE_FACTOR", "SUP scale factor l", 1.0);
	/* 1*/ AddParameter("SURFACE_ENERGY", "Surface energy density [J/m²]", 0.0);
}

void CModelPPSUP::CalculatePP(double _time, double _timeStep, size_t _iSrc, size_t _iDst, 
                               const SInteractProps& _interactProp, SCollision* _collision) const
{
	// 获取SUP参数
	const double l = m_parameters[0].value;      // SUP缩放因子
	const double gamma = m_parameters[1].value;  // 表面能密度 (JKR粘附)
	
	// 获取粒子属性
	const CVector3 anglVel1 = Particles().AnglVel(_iSrc);
	const CVector3 anglVel2 = Particles().AnglVel(_iDst);
	const double   radius1  = Particles().Radius(_iSrc);
	const double   radius2  = Particles().Radius(_iDst);
	
	// 计算接触向量
	const CVector3 rc1        = _collision->vContactVector * (radius1 / (radius1 + radius2));
	const CVector3 rc2        = _collision->vContactVector * (-radius2 / (radius1 + radius2));
	const CVector3 normVector = _collision->vContactVector.Normalized();
	
	// 相对速度计算
	const CVector3 relVel        = (Particles().Vel(_iDst) + anglVel2 * rc2) - 
																	(Particles().Vel(_iSrc) + anglVel1 * rc1);
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;
	
	// 接触区域半径（基于实际重叠）
	const double contactAreaRadius = std::sqrt(_collision->dEquivRadius * _collision->dNormalOverlap);
	
	// SUP缩放的法向刚度
	const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius * l;
	
	// 计算法向力（Hertz-Mindlin + JKR粘附）
	double normContactForceLen;
    
	if (gamma > 0) {
		// JKR粘附力（SUP缩放）
		const double a3 = std::pow(contactAreaRadius, 3.0);
		const double elasticForce = 4.0 * a3 * _interactProp.dEquivYoungModulus * l / 
																	(3.0 * _collision->dEquivRadius);
		const double adhesionForce = std::sqrt(8 * PI * _interactProp.dEquivYoungModulus * l * 
																						gamma * l * l * a3);  // γ缩放为l²
		normContactForceLen = -1.0 * (elasticForce - adhesionForce);
	} else {
		// 纯Hertz-Mindlin（无粘附）
		normContactForceLen = 2.0 / 3.0 * _collision->dNormalOverlap * Kn;
	}
    
	// 法向阻尼力
	const double normDampingForceLen = -_2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * 
																			std::sqrt(Kn * _collision->dEquivMass);
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
	const CVector3 tangShearForce = tangOverlap * Kt;
	const CVector3 tangDampingForce = tangRelVel * 
																		(-_2_SQRT_5_6 * _interactProp.dAlpha * 
																			std::sqrt(Kt * _collision->dEquivMass));
	
	// 检查滑动条件
	CVector3 tangForce;
	const double tangShearForceLen = tangShearForce.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * 
																	std::abs(normContactForceLen + normDampingForceLen);
	
	if (tangShearForceLen > frictionForceLen) {
		tangForce = tangShearForce * frictionForceLen / tangShearForceLen;
		tangOverlap = tangForce / Kt;
	} else {
		tangForce = tangShearForce + tangDampingForce;
	}
    
	// 滚动阻力（SUP缩放）
	const CVector3 rollingTorque1 = anglVel1.IsSignificant() ? 
			anglVel1 * (-_interactProp.dRollingFriction * std::abs(normContactForceLen) * 
									radius1 / anglVel1.Length()) : CVector3{0};
	const CVector3 rollingTorque2 = anglVel2.IsSignificant() ? 
			anglVel2 * (-_interactProp.dRollingFriction * std::abs(normContactForceLen) * 
									radius2 / anglVel2.Length()) : CVector3{0};
    
	// 应用SUP缩放到最终的力和力矩
	const CVector3 totalForce = (normForce + tangForce) * l * l;  // 力缩放 l²
	const CVector3 moment1 = (normVector * tangForce * radius1 + rollingTorque1) * l * l * l;  // 力矩缩放 l³
	const CVector3 moment2 = (normVector * tangForce * radius2 + rollingTorque2) * l * l * l;
	
	// 存储结果
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce * l * l;  // 存储缩放后的切向力
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment1;
	_collision->vResultMoment2 = moment2;
}

void CModelPPSUP::ConsolidateSrc(double _time, double _timeStep, size_t _iPart, 
                                  SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart)  += _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment1;
}

void CModelPPSUP::ConsolidateDst(double _time, double _timeStep, size_t _iPart, 
                                  SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart)  -= _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment2;
}