/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPPopovJKR.h"

CModelPPPopovJKR::CModelPPPopovJKR()
{
	m_name         = "SUP-JKR";
	m_uniqueKey    = "7A4C0DAF1F6D44AE925F4DD84563E36F";
	m_helpFileName = "/Contact Models/PopovJKR.pdf";
	m_hasGPUSupport = true;
    
	// SUP模型参数
	/* 0*/ AddParameter("SCALE_FACTOR", "SUP scale factor l", 1.0);
}

void CModelPPPopovJKR::CalculatePP(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// 获取SUP缩放因子
	const double l = m_parameters[0].value;
	
	// ========== 第一步：从放大颗粒参数转换到原始颗粒参数 ==========
	
	// 1. 几何参数转换
	const double radius1_S = Particles().Radius(_iSrc);  // 放大半径
	const double radius2_S = Particles().Radius(_iDst);
	const double radius1_O = radius1_S / l;  // 原始半径
	const double radius2_O = radius2_S / l;
	
	// 2. 重叠量转换：δ_O = δ_S / l
	const double overlap_S = _collision->dNormalOverlap;
	const double overlap_O = overlap_S / l;
	const double equivRadius_S = _collision->dEquivRadius;
	const double equivRadius_O = equivRadius_S / l;
	const double equivMass_O = _collision->dEquivMass / (l * l * l);  // 质量缩放 m_O = m_S/l³
	
	// 3. 角速度转换：ω_O = l × ω_S
	const CVector3 anglVel1_S = Particles().AnglVel(_iSrc);
	const CVector3 anglVel2_S = Particles().AnglVel(_iDst);
	const CVector3 anglVel1_O = anglVel1_S * l;
	const CVector3 anglVel2_O = anglVel2_S * l;
	
	// 4. 接触向量（注意：这里也需要转换到原始尺度）
	const CVector3 contactVector_O = _collision->vContactVector / l;
	const CVector3 rc1_O = contactVector_O * (radius1_O / (radius1_O + radius2_O));
	const CVector3 rc2_O = contactVector_O * (-radius2_O / (radius1_O + radius2_O));
	const CVector3 normVector = _collision->vContactVector.Normalized();
	
	// 5. 相对速度计算（使用原始参数）
	const CVector3 relVel = (Particles().Vel(_iDst) + anglVel2_O * rc2_O) - 
	                        (Particles().Vel(_iSrc) + anglVel1_O * rc1_O);
	const double normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel = normRelVelLen * normVector;
	const CVector3 tangRelVel = relVel - normRelVel;
	
	// ========== 第二步：使用原始参数计算原始颗粒的力和力矩 ==========
	
	// 1. 接触区域半径（基于原始重叠）
	const double contactAreaRadius_O = std::sqrt(equivRadius_O * overlap_O);
	
	// 2. 法向刚度（原始颗粒）
	const double Kn_O = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius_O;
	
	// 3. 法向力（原始颗粒）
	double normContactForceLen_O;
	if (_interactProp.dEquivSurfaceEnergy > 0) {
		// JKR adhesion model
		const double a3_O = std::pow(contactAreaRadius_O, 3.0);
		const double elasticForce = 4.0 * a3_O * _interactProp.dEquivYoungModulus / (3.0 * equivRadius_O);
		const double adhesionForce = std::sqrt(8 * PI * _interactProp.dEquivYoungModulus * 
		                                       _interactProp.dEquivSurfaceEnergy * a3_O);
		normContactForceLen_O = -1.0 * (elasticForce - adhesionForce);
	} else {
		normContactForceLen_O = -2.0 / 3.0 * overlap_O * Kn_O;
	}
	
	const double normDampingForceLen_O = -_2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * 
	                                     std::sqrt(Kn_O * equivMass_O);
	const CVector3 normForce_O = normVector * (normContactForceLen_O + normDampingForceLen_O);
	
	// 4. 切向力（原始颗粒）
	// 旋转旧的切向重叠
	CVector3 tangOverlapRot = _collision->vTangOverlap - normVector * DotProduct(normVector, _collision->vTangOverlap);
	if (tangOverlapRot.IsSignificant())
		tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
	
	// 计算新的切向重叠（使用原始时间步长）
	// const double timeStep_O = _timeStep / l;  // 时间步长缩放：Δt_O = Δt_S / l
	const double timeStep_O = _timeStep;  // 暂时不缩放时间步长
	CVector3 tangOverlap_O = tangOverlapRot / l + tangRelVel * timeStep_O;  // 转换到原始尺度
	
	// 切向刚度（原始颗粒）
	const double Kt_O = 8 * _interactProp.dEquivShearModulus * contactAreaRadius_O;
	const CVector3 tangShearForce_O = tangOverlap_O * Kt_O;
	const CVector3 tangDampingForce_O = tangRelVel * (-_2_SQRT_5_6 * _interactProp.dAlpha * 
	                                                   std::sqrt(Kt_O * equivMass_O));
	
	// 检查滑动条件
	CVector3 tangForce_O;
	const double tangShearForceLen = tangShearForce_O.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * 
	                                std::abs(normContactForceLen_O + normDampingForceLen_O);
	
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
		anglVel1_O * (-_interactProp.dRollingFriction * std::abs(normContactForceLen_O) * 
		              radius1_O / anglVel1_O.Length()) : CVector3{0};
	const CVector3 rollingTorque2_O = anglVel2_O.IsSignificant() ? 
		anglVel2_O * (-_interactProp.dRollingFriction * std::abs(normContactForceLen_O) * 
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
	_collision->vTangOverlap = tangOverlap_O * l;  // 转换回放大尺度存储
	_collision->vTangForce = tangForce_O * l * l;  // 存储缩放后的切向力
	_collision->vTotalForce = totalForce_S;
	_collision->vResultMoment1 = moment1_S;
	_collision->vResultMoment2 = moment2_S;
}

void CModelPPPopovJKR::ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart)  += _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment1;
}

void CModelPPPopovJKR::ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart)  -= _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment2;
}
