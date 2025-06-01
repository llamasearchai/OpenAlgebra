# OpenAlgebra Medical AI - Operational Runbook

## Table of Contents
1. [System Overview](#system-overview)
2. [Deployment Procedures](#deployment-procedures)
3. [Monitoring & Alerts](#monitoring--alerts)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Emergency Procedures](#emergency-procedures)
6. [Maintenance Tasks](#maintenance-tasks)
7. [Security Procedures](#security-procedures)
8. [Compliance Checks](#compliance-checks)

## System Overview

OpenAlgebra Medical AI is a high-performance sparse linear algebra system designed for medical AI applications. The system processes medical imaging data (DICOM, NIfTI) and provides AI-powered analysis for various medical specialties.

### Architecture Components
- **Core Service**: Rust-based processing engine
- **API Gateway**: FastAPI endpoints for medical AI services
- **Database**: PostgreSQL for metadata and results
- **Cache**: Redis for session management and caching
- **Object Storage**: MinIO for medical images and models
- **GPU Compute**: CUDA/Metal acceleration for AI inference

## Deployment Procedures

### Initial Deployment

1. **Pre-deployment Checklist**
   ```bash
   # Verify environment variables
   ./scripts/check-env.sh
   
   # Run pre-deployment tests
   cargo test --release
   
   # Verify HIPAA compliance
   ./scripts/hipaa-check.sh
   ```

2. **Deploy with Docker Compose**
   ```bash
   # Production deployment
   ./scripts/deploy.sh latest production yourdomain.com
   
   # Verify deployment
   curl https://yourdomain.com/health
   ```

3. **Post-deployment Verification**
   - Check all services are running: `docker-compose ps`
   - Verify API endpoints: `./scripts/smoke-test.sh`
   - Check logs: `docker-compose logs -f`

### Rolling Updates

1. **Build new version**
   ```bash
   cargo build --release
   docker build -t openalgebra/medical-ai:v1.2.3 .
   ```

2. **Blue-Green Deployment**
   ```bash
   # Deploy to staging
   ./scripts/deploy.sh v1.2.3 staging staging.yourdomain.com
   
   # Run integration tests
   cargo test --features integration
   
   # Switch production
   ./scripts/blue-green-switch.sh production v1.2.3
   ```

## Monitoring & Alerts

### Key Metrics to Monitor

1. **System Health**
   - CPU Usage: Alert if > 80% for 5 minutes
   - Memory Usage: Alert if > 85% for 5 minutes
   - Disk Usage: Alert if > 90%
   - GPU Utilization: Alert if < 30% during processing

2. **Application Metrics**
   - API Response Time: Alert if p95 > 1s
   - Error Rate: Alert if > 1% for 5 minutes
   - Model Accuracy: Alert if < 90%
   - DICOM Processing Time: Alert if > 10s

3. **Compliance Metrics**
   - Unanonymized Data: Critical alert if any detected
   - Audit Log Gaps: Alert if logging fails
   - Encryption Status: Alert if disabled

### Alert Response Procedures

1. **High Error Rate**
   ```bash
   # Check recent errors
   docker-compose logs openalgebra --tail 1000 | grep ERROR
   
   # Check API health
   curl localhost:8000/health
   
   # Restart if necessary
   docker-compose restart openalgebra
   ```

2. **Memory Pressure**
   ```bash
   # Check memory usage
   docker stats
   
   # Clear cache if needed
   docker-compose exec redis redis-cli FLUSHDB
   
   # Scale horizontally
   docker-compose up -d --scale openalgebra=3
   ```

## Common Issues & Solutions

### Issue: DICOM Processing Timeout

**Symptoms**: DICOM files taking > 30s to process

**Solution**:
1. Check file size: `ls -lh /data/dicom/`
2. Verify GPU acceleration: `nvidia-smi`
3. Increase timeout: `DICOM_TIMEOUT=60 docker-compose up -d`

### Issue: Model Accuracy Degradation

**Symptoms**: Accuracy drops below 90%

**Solution**:
1. Check recent model updates: `git log --oneline models/`
2. Validate test dataset: `cargo test test_model_accuracy`
3. Rollback if needed: `./scripts/model-rollback.sh previous_version`

### Issue: Database Connection Errors

**Symptoms**: "connection refused" errors

**Solution**:
1. Check PostgreSQL status: `docker-compose ps postgres`
2. Verify connections: `docker-compose exec postgres pg_isready`
3. Reset connections: `docker-compose restart postgres openalgebra`

## Emergency Procedures

### Critical Patient Data Exposure

1. **Immediate Actions**
   ```bash
   # Stop all processing
   docker-compose stop openalgebra openalgebra-gpu
   
   # Enable emergency mode
   echo "EMERGENCY_MODE=true" >> .env
   docker-compose up -d nginx  # Only static responses
   ```

2. **Investigation**
   ```bash
   # Check access logs
   grep -E "patient|ssn|name" /var/log/openalgebra/access.log
   
   # Generate compliance report
   ./scripts/generate-hipaa-report.sh
   ```

3. **Recovery**
   ```bash
   # Clear sensitive data from logs
   ./scripts/sanitize-logs.sh
   
   # Re-enable services
   echo "EMERGENCY_MODE=false" >> .env
   docker-compose up -d
   ```

### Complete System Failure

1. **Failover to Backup**
   ```bash
   # Switch DNS to backup site
   ./scripts/dns-failover.sh backup.yourdomain.com
   
   # Restore from latest backup
   ./scripts/restore-backup.sh latest
   ```

2. **Root Cause Analysis**
   - Collect all logs: `./scripts/collect-diagnostics.sh`
   - Review monitoring data from last 24h
   - Document timeline in incident report

## Maintenance Tasks

### Daily Tasks
- [ ] Check backup completion
- [ ] Review error logs
- [ ] Verify DICOM processing queue
- [ ] Check disk usage

### Weekly Tasks
- [ ] Update virus definitions
- [ ] Review access logs for anomalies
- [ ] Test backup restoration
- [ ] Update SSL certificates if needed

### Monthly Tasks
- [ ] Security patches
- [ ] Performance analysis
- [ ] Compliance audit
- [ ] Disaster recovery drill

## Security Procedures

### Access Control

1. **Adding New User**
   ```bash
   # Generate secure credentials
   ./scripts/create-user.sh username role
   
   # Configure RBAC
   kubectl apply -f k8s/rbac/user-role.yaml
   ```

2. **Rotating Secrets**
   ```bash
   # Rotate all secrets
   ./scripts/rotate-secrets.sh
   
   # Update applications
   docker-compose down
   docker-compose up -d
   ```

### Security Incident Response

1. **Detection**
   - Monitor IDS alerts
   - Check failed login attempts
   - Review API access patterns

2. **Containment**
   ```bash
   # Block suspicious IP
   iptables -A INPUT -s SUSPICIOUS_IP -j DROP
   
   # Revoke compromised tokens
   ./scripts/revoke-tokens.sh
   ```

3. **Recovery**
   - Reset affected credentials
   - Review and patch vulnerabilities
   - Update security policies

## Compliance Checks

### HIPAA Compliance

**Daily Checks**:
```bash
# Verify encryption
./scripts/check-encryption.sh

# Audit data access
./scripts/audit-access.sh --last 24h

# Check anonymization
./scripts/verify-anonymization.sh
```

**Monthly Audit**:
```bash
# Full compliance scan
./scripts/hipaa-compliance-scan.sh --full

# Generate report
./scripts/generate-compliance-report.sh --format pdf
```

### FDA 510(k) Preparation

**Validation Procedures**:
```bash
# Run clinical validation suite
cargo test --features clinical_validation

# Generate validation report
./scripts/generate-510k-docs.sh
```

**Documentation Requirements**:
- Software architecture document
- Risk analysis (ISO 14971)
- Clinical validation data
- User manual
- Change control procedures

## Contact Information

### Escalation Path
1. **Level 1**: On-call Engineer - oncall@openalgebra.ai
2. **Level 2**: Team Lead - teamlead@openalgebra.ai
3. **Level 3**: CTO - cto@openalgebra.ai

### External Contacts
- **AWS Support**: [Support Case URL]
- **Security Team**: security@openalgebra.ai
- **Compliance Officer**: compliance@openalgebra.ai
- **Legal Team**: legal@openalgebra.ai

### Vendor Support
- **NVIDIA GPU Issues**: gpu-support@nvidia.com
- **PostgreSQL**: enterprise-support@postgresql.org
- **Docker**: support@docker.com

---

Last Updated: 2024-01-15
Version: 1.0.0
Next Review: 2024-02-15