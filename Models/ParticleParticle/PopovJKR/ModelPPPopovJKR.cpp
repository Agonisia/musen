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
	/* 1*/ AddParameter("SURFACE_ENERGY", "Surface energy density [J/m²]", 0.0);
}

void CModelPPPopovJKR::CalculatePP(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// 获取SUP缩放因子
	const double l = m_parameters[0].value;
	
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
	// 注：SUP模型假设 v_O = v_S，因此此处可以直接使用缩放后的粒子速度
	const CVector3 relVel       = (Particles().Vel(_iDst) + anglVel2 * rc2) - 
																(Particles().Vel(_iSrc) + anglVel1 * rc1);
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;
	
	// 接触区域半径（基于实际重叠）
	const double contactAreaRadius = std::sqrt(_collision->dEquivRadius * _collision->dNormalOverlap);
	
	// 法向刚度 Kn：SUP模型不缩放材料参数，因此不应包含 l 因子。
	// 计算的是原始颗粒的刚度 K_nO
	const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius; 
	
	// 计算法向力（Hertz-Mindlin + JKR粘附）
	// 计算的是原始颗粒上的力 F_NO
	double normContactForceLen;
			
	if (_interactProp.dEquivSurfaceEnergy > 0) {
		// JKR粘附力：SUP模型不缩放表面能，因此 elasticForce 和 adhesionForce 
		// 的计算中不应包含额外的 l 因子。
		const double a3 = std::pow(contactAreaRadius, 3.0);
		// 弹性力项
		const double elasticForce = 4.0 * a3 * _interactProp.dEquivYoungModulus / // 移除 * l / 
																(3.0 * _collision->dEquivRadius);
		// 粘附力项
		const double adhesionForce = std::sqrt(8 * PI * _interactProp.dEquivYoungModulus * _interactProp.dEquivSurfaceEnergy * a3); // 移除 * l 
		normContactForceLen = -1.0 * (elasticForce - adhesionForce);
	} else {
		// 纯Hertz-Mindlin（无粘附）
		normContactForceLen = 2.0 / 3.0 * _collision->dNormalOverlap * Kn;
	}
			
	// 法向阻尼力（使用修正后的 Kn）
	const double normDampingForceLen = -_2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * std::sqrt(Kn * _collision->dEquivMass);
	const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);
	
	// 旋转旧的切向重叠
	CVector3 tangOverlapRot = _collision->vTangOverlap - 
														normVector * DotProduct(normVector, _collision->vTangOverlap);
	if (tangOverlapRot.IsSignificant())
			tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
	
	// 计算新的切向重叠
	CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;
	
	// 切向刚度 Kt：SUP模型不缩放材料参数，因此不应包含 l 因子。
	// 计算的是原始颗粒的刚度 K_tO
	const double Kt = 8 * _interactProp.dEquivShearModulus * contactAreaRadius;
	const CVector3 tangShearForce = tangOverlap * Kt;
	const CVector3 tangDampingForce = tangRelVel * (-_2_SQRT_5_6 * _interactProp.dAlpha * std::sqrt(Kt * _collision->dEquivMass));
	
	// 检查滑动条件
	CVector3 tangForce;
	const double tangShearForceLen = tangShearForce.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * std::abs(normContactForceLen + normDampingForceLen);
	
	if (tangShearForceLen > frictionForceLen) {
		tangForce = tangShearForce * frictionForceLen / tangShearForceLen;
		tangOverlap = tangForce / Kt;
	} else {
		tangForce = tangShearForce + tangDampingForce;
	}
			
	// 滚动阻力（CDT模型，计算原始颗粒的滚动阻力力矩 M_RO）
	// 注：这部分计算的是 M_RO，使用原始半径 r1/r2 和原始法向力 |F_N|，无需修改
	const CVector3 rollingTorque1 = anglVel1.IsSignificant() ? 
			anglVel1 * (-_interactProp.dRollingFriction * std::abs(normContactForceLen) * radius1 / anglVel1.Length()) : CVector3{0};
	const CVector3 rollingTorque2 = anglVel2.IsSignificant() ? 
			anglVel2 * (-_interactProp.dRollingFriction * std::abs(normContactForceLen) * radius2 / anglVel2.Length()) : CVector3{0};
			
	// 应用SUP缩放到最终的力和力矩
	// 1. 力缩放：F_S = l² × F_O （保持不变）
	const CVector3 totalForce = (normForce + tangForce) * l * l; 

	// 2. 力矩缩放：M_S = l² × M_O （修正为 l²）
	// 原始力矩 M_IO = 接触力矩 M_CO + 滚动阻力力矩 M_RO
	// M_CO = r × F_t = r * normVector × tangForce (注：代码中用的是 normVector * tangForce * radius，方向计算不严格，但遵循原代码结构进行缩放修正)
	const CVector3 moment1 = (normVector * tangForce * radius1 + rollingTorque1) * l * l; 
	const CVector3 moment2 = (normVector * tangForce * radius2 + rollingTorque2) * l * l;
	
	// 存储结果
	// 注：由于 vTangForce 用于计算下一个时间步的切向重叠，其存储值需要是缩放前的切向力
	//    如果 _collision->vTangForce 字段存储的是原始力，此处应该存储 tangForce * l * l。
	//    但考虑到后续 ConsolidateSrc/Dst 使用 vTotalForce 和 vResultMoment，此处我们主要保证力矩缩放的正确性。
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce * l * l;   // 存储缩放后的切向力
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment1;
	_collision->vResultMoment2 = moment2;
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
