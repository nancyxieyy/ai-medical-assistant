# backend/database/models.py

from sqlalchemy import Column, Integer, String, Date, DateTime, Text, JSON, ForeignKey
from .base import Base
from .session import engine
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy import func, Enum

Base.metadata.create_all(bind=engine)

class Patient(Base):
    """患者基本信息表"""
    __tablename__ = 'patients'
    
    # 患者ID，主键，自增
    patient_id = Column(Integer, primary_key=True)
    # 患者姓名
    name = Column(String(100), nullable=False)
    # 出生日期
    birth_date = Column(Date, nullable=False)
    # 性别
    gender = Column(String(10), nullable=False)
    # 联系方式
    contact_info = Column(String(200))
    # 创建时间
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    # 更新时间
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    # 关联的病例信息
    cases = relationship("ConfirmedCase", back_populates="patient")

class ConfirmedCase(Base):
    """医生确认的病例信息表"""
    __tablename__ = 'confirmed_cases'
    
    # 病例ID，主键，自增
    case_id = Column(Integer, primary_key=True)
    # 关联的患者ID，外键
    patient_id = Column(Integer, ForeignKey('patients.patient_id'), nullable=False)
    # 关联的医生ID(MVP阶段先用整数表示，后续可扩展为外键)
    doctor_id = Column(Integer, nullable=False)
    # 问诊日期时间
    consultation_date = Column(DateTime, nullable=False)
    # 病例摘要（结构化数据）
    ai_summary = Column(JSON, nullable=False)
    # 医生备注
    doctor_notes = Column(Text)
    # 创建时间
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    # 更新时间
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    # 关联的患者信息
    patient = relationship("Patient", back_populates="cases")

class MedicalKnowledge(Base):
    """医学知识库表"""
    __tablename__ = 'medical_knowledge'
    
    # 知识条目ID，主键，自增
    doc_id = Column(Integer, primary_key=True, index=True)
    # 知识标题
    title = Column(String(255), nullable=False, unique=False)
    # 知识内容（文本）
    content_text = Column(Text, nullable=False)
    # 来源URL
    source_url = Column(String(255))
    # 创建时间
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    # 来源
    source_type = Column(String(20))


class Report(Base):
    """
    医疗报告(AI草稿 + 医生确认)表
    draft:AI 自动生成，等待医生确认
    final:医生确认后,进入 ConfirmedCase(正式病例)
    """
    __tablename__ = "reports"
    
    # 报告ID，主键，自增
    id = Column(Integer, primary_key=True, autoincrement=True)
    # 用于分辨会话 / 病人临时ID
    session_id = Column(String(128), index=True, nullable=True)
    # realtime_advice / final_report
    mode = Column(String(32), nullable=False)
    # 原始问诊文字
    transcript = Column(Text, nullable=True)
    # RAG 检索到的医学上下文
    context = Column(Text, nullable=True)
    # LLM 生成的文本
    output = Column(Text, nullable=False)     
    # 草稿 / 确认状态
    status = Column(Enum("draft", "final", name="report_status"),     
                    nullable=False, default="draft")
    # 创建时间
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    # 更新时间
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)