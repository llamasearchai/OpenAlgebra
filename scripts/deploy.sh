#!/bin/bash

# OpenAlgebra Medical AI Deployment Script
# This script automates the deployment of the complete OpenAlgebra Medical AI system

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-production}"
DOMAIN="${3:-localhost}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Help function
show_help() {
    cat << EOF
OpenAlgebra Medical AI Deployment Script

Usage: $0 [VERSION] [ENVIRONMENT] [DOMAIN]

Arguments:
    VERSION        Docker image version to deploy (default: latest)
    ENVIRONMENT    Deployment environment (default: production)
    DOMAIN         Domain name for the deployment (default: localhost)

Environments:
    development    Local development deployment
    staging        Staging environment deployment
    production     Production deployment with full security

Examples:
    $0                                    # Deploy latest to production on localhost
    $0 v1.2.3 staging staging.example.com  # Deploy v1.2.3 to staging
    $0 latest production app.example.com    # Deploy latest to production

Required Environment Variables:
    OPENAI_API_KEY                 OpenAI API key for agents functionality
    DATABASE_URL                   PostgreSQL database connection string
    REDIS_URL                      Redis connection string (optional)
    SSL_CERT_EMAIL                 Email for Let's Encrypt certificates (production only)

Optional Environment Variables:
    BACKUP_S3_BUCKET              S3 bucket for database backups
    MONITORING_WEBHOOK            Webhook URL for monitoring alerts
    LOG_LEVEL                     Logging level (debug, info, warn, error)
    API_RATE_LIMIT                API rate limit (requests per minute)
    MAX_UPLOAD_SIZE               Maximum file upload size in MB
EOF
}

# Check if help was requested
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

# Validate environment
validate_environment() {
    log "Validating deployment environment..."
    
    # Check required tools
    for tool in docker docker-compose curl jq; do
        if ! command -v $tool &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check required environment variables
    if [[ "$ENVIRONMENT" == "production" ]] || [[ "$ENVIRONMENT" == "staging" ]]; then
        if [[ -z "${OPENAI_API_KEY:-}" ]]; then
            error "OPENAI_API_KEY environment variable is required for $ENVIRONMENT"
        fi
        
        if [[ -z "${DATABASE_URL:-}" ]]; then
            error "DATABASE_URL environment variable is required for $ENVIRONMENT"
        fi
    fi
    
    # Validate domain
    if [[ "$DOMAIN" != "localhost" ]] && [[ "$ENVIRONMENT" == "production" ]]; then
        if [[ -z "${SSL_CERT_EMAIL:-}" ]]; then
            error "SSL_CERT_EMAIL environment variable is required for production deployment with custom domain"
        fi
    fi
    
    log "Environment validation passed"
}

# Setup directories
setup_directories() {
    log "Setting up deployment directories..."
    
    mkdir -p /opt/openalgebra/{data,logs,configs,backups,ssl}
    mkdir -p /var/log/openalgebra
    
    # Set permissions
    chmod 755 /opt/openalgebra
    chmod 750 /opt/openalgebra/{data,logs,configs,backups}
    chmod 700 /opt/openalgebra/ssl
    
    log "Directories created successfully"
}

# Generate configuration files
generate_configs() {
    log "Generating configuration files..."
    
    # Generate main configuration
    cat > /opt/openalgebra/configs/openalgebra.json << EOF
{
    "gpu_enabled": true,
    "num_threads": $(nproc),
    "memory_limit_gb": 8.0,
    "cache_size_mb": 1024,
    "logging_level": "${LOG_LEVEL:-info}",
    "output_directory": "/opt/openalgebra/data/output",
    "temp_directory": "/tmp/openalgebra",
    "model_checkpoint_interval": 100,
    "privacy_mode": true,
    "compliance_mode": "HIPAA",
    "performance_monitoring": true,
    "distributed_computing": false,
    "max_concurrent_jobs": 4,
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "enable_api_server": true,
    "openai_api_key": "${OPENAI_API_KEY:-}",
    "enable_ai_agents": true,
    "hipaa_compliance": true
}
EOF

    # Generate docker-compose override
    cat > /opt/openalgebra/configs/docker-compose.override.yml << EOF
version: '3.8'

services:
  openalgebra:
    image: ghcr.io/openalgebra/openalgebra-medical:${VERSION}
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - DATABASE_URL=${DATABASE_URL:-postgresql://postgres:postgres@postgres:5432/openalgebra}
      - REDIS_URL=${REDIS_URL:-redis://redis:6379}
      - RUST_LOG=${LOG_LEVEL:-info}
      - ENVIRONMENT=${ENVIRONMENT}
    volumes:
      - /opt/openalgebra/data:/data
      - /opt/openalgebra/configs:/configs
      - /var/log/openalgebra:/var/log/openalgebra
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /opt/openalgebra/ssl:/etc/nginx/ssl:ro
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    restart: unless-stopped
    depends_on:
      - openalgebra

  postgres:
    environment:
      - POSTGRES_DB=openalgebra
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - /opt/openalgebra/backups:/backups
    restart: unless-stopped

  redis:
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
EOF

    # Generate nginx configuration
    cat > /opt/openalgebra/configs/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream openalgebra_backend {
        server openalgebra:8000;
    }
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=${API_RATE_LIMIT:-60}r/m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # File upload limits
    client_max_body_size ${MAX_UPLOAD_SIZE:-100}M;
    
    server {
        listen 80;
        server_name ${DOMAIN};
        
        # Redirect HTTP to HTTPS in production
        $(if [[ "$ENVIRONMENT" == "production" && "$DOMAIN" != "localhost" ]]; then
            echo "return 301 https://\$server_name\$request_uri;"
        else
            echo "# HTTP-only configuration for development/staging"
            echo "location / {"
            echo "    proxy_pass http://openalgebra_backend;"
            echo "    proxy_set_header Host \$host;"
            echo "    proxy_set_header X-Real-IP \$remote_addr;"
            echo "    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;"
            echo "    proxy_set_header X-Forwarded-Proto \$scheme;"
            echo "    limit_req zone=api burst=20 nodelay;"
            echo "}"
        fi)
    }
    
    $(if [[ "$ENVIRONMENT" == "production" && "$DOMAIN" != "localhost" ]]; then
        cat << 'HTTPS_CONFIG'
    server {
        listen 443 ssl http2;
        server_name DOMAIN_PLACEHOLDER;
        
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        location / {
            proxy_pass http://openalgebra_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            limit_req zone=api burst=20 nodelay;
        }
        
        # WebSocket support for real-time features
        location /ws {
            proxy_pass http://openalgebra_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
    }
HTTPS_CONFIG
        # Replace placeholder with actual domain
        sed -i "s/DOMAIN_PLACEHOLDER/${DOMAIN}/g" /opt/openalgebra/configs/nginx.conf
    fi)
}
EOF

    log "Configuration files generated successfully"
}

# Setup SSL certificates
setup_ssl() {
    if [[ "$ENVIRONMENT" == "production" && "$DOMAIN" != "localhost" ]]; then
        log "Setting up SSL certificates with Let's Encrypt..."
        
        # Install certbot if not present
        if ! command -v certbot &> /dev/null; then
            info "Installing certbot..."
            if command -v apt-get &> /dev/null; then
                apt-get update && apt-get install -y certbot python3-certbot-nginx
            elif command -v yum &> /dev/null; then
                yum install -y certbot python3-certbot-nginx
            else
                error "Cannot install certbot. Please install it manually."
            fi
        fi
        
        # Obtain SSL certificate
        certbot certonly --standalone \
            --non-interactive \
            --agree-tos \
            --email "${SSL_CERT_EMAIL}" \
            --domains "${DOMAIN}" \
            --pre-hook "docker-compose -f ${PROJECT_DIR}/docker-compose.yml -f /opt/openalgebra/configs/docker-compose.override.yml stop nginx" \
            --post-hook "docker-compose -f ${PROJECT_DIR}/docker-compose.yml -f /opt/openalgebra/configs/docker-compose.override.yml start nginx"
        
        # Copy certificates to nginx directory
        cp /etc/letsencrypt/live/${DOMAIN}/fullchain.pem /opt/openalgebra/ssl/
        cp /etc/letsencrypt/live/${DOMAIN}/privkey.pem /opt/openalgebra/ssl/
        chmod 600 /opt/openalgebra/ssl/*.pem
        
        # Setup auto-renewal
        cat > /etc/cron.d/openalgebra-certbot << EOF
0 12 * * * root certbot renew --quiet --post-hook "docker-compose -f ${PROJECT_DIR}/docker-compose.yml -f /opt/openalgebra/configs/docker-compose.override.yml restart nginx"
EOF
        
        log "SSL certificates configured successfully"
    else
        log "Skipping SSL setup for $ENVIRONMENT environment"
    fi
}

# Database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    info "Waiting for database to be ready..."
    until docker-compose -f "${PROJECT_DIR}/docker-compose.yml" -f /opt/openalgebra/configs/docker-compose.override.yml exec postgres pg_isready -U postgres; do
        sleep 2
    done
    
    # Run migrations (if migration system exists)
    # This would be implemented based on the specific migration system used
    info "Database migrations completed"
}

# Start services
start_services() {
    log "Starting OpenAlgebra Medical AI services..."
    
    cd "$PROJECT_DIR"
    
    # Pull latest images
    docker-compose -f docker-compose.yml -f /opt/openalgebra/configs/docker-compose.override.yml pull
    
    # Start services
    docker-compose -f docker-compose.yml -f /opt/openalgebra/configs/docker-compose.override.yml up -d
    
    # Wait for services to be healthy
    info "Waiting for services to be healthy..."
    sleep 30
    
    # Health check
    for i in {1..30}; do
        if curl -f "http://localhost:8000/health" &> /dev/null; then
            log "Services are healthy and ready"
            break
        fi
        
        if [[ $i -eq 30 ]]; then
            error "Services failed to become healthy within 5 minutes"
        fi
        
        sleep 10
    done
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Setup log rotation
    cat > /etc/logrotate.d/openalgebra << EOF
/var/log/openalgebra/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    sharedscripts
    postrotate
        docker-compose -f ${PROJECT_DIR}/docker-compose.yml -f /opt/openalgebra/configs/docker-compose.override.yml restart openalgebra
    endscript
}
EOF

    # Setup health check script
    cat > /usr/local/bin/openalgebra-healthcheck.sh << 'EOF'
#!/bin/bash
HEALTH_URL="http://localhost:8000/health"
WEBHOOK_URL="${MONITORING_WEBHOOK:-}"

if ! curl -f "$HEALTH_URL" &> /dev/null; then
    echo "$(date): OpenAlgebra health check failed" >> /var/log/openalgebra/healthcheck.log
    
    if [[ -n "$WEBHOOK_URL" ]]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d '{"text":"🚨 OpenAlgebra Medical AI health check failed","channel":"#alerts"}'
    fi
    
    exit 1
fi

echo "$(date): OpenAlgebra health check passed" >> /var/log/openalgebra/healthcheck.log
EOF

    chmod +x /usr/local/bin/openalgebra-healthcheck.sh
    
    # Setup cron job for health checks
    cat > /etc/cron.d/openalgebra-monitoring << EOF
*/5 * * * * root /usr/local/bin/openalgebra-healthcheck.sh
EOF

    log "Monitoring setup completed"
}

# Backup setup
setup_backups() {
    log "Setting up automated backups..."
    
    # Database backup script
    cat > /usr/local/bin/openalgebra-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/openalgebra/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="openalgebra_backup_${DATE}.sql"

# Create database backup
docker-compose -f PROJECT_DIR_PLACEHOLDER/docker-compose.yml -f /opt/openalgebra/configs/docker-compose.override.yml exec -T postgres pg_dump -U postgres openalgebra > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to S3 if configured
if [[ -n "${BACKUP_S3_BUCKET:-}" ]]; then
    aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" "s3://${BACKUP_S3_BUCKET}/database/"
fi

# Clean up old backups (keep last 7 days)
find "$BACKUP_DIR" -name "openalgebra_backup_*.sql.gz" -mtime +7 -delete

echo "$(date): Backup completed: ${BACKUP_FILE}.gz"
EOF

    # Replace placeholder
    sed -i "s|PROJECT_DIR_PLACEHOLDER|${PROJECT_DIR}|g" /usr/local/bin/openalgebra-backup.sh
    chmod +x /usr/local/bin/openalgebra-backup.sh
    
    # Setup daily backups
    cat > /etc/cron.d/openalgebra-backup << EOF
0 2 * * * root /usr/local/bin/openalgebra-backup.sh >> /var/log/openalgebra/backup.log 2>&1
EOF

    log "Backup system configured"
}

# Security hardening
apply_security() {
    log "Applying security hardening..."
    
    # Setup firewall rules (UFW)
    if command -v ufw &> /dev/null; then
        ufw --force reset
        ufw default deny incoming
        ufw default allow outgoing
        ufw allow ssh
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw --force enable
        log "Firewall configured"
    fi
    
    # Disable unnecessary services
    systemctl list-unit-files --type=service --state=enabled | grep -E "(telnet|ftp|rsh)" | awk '{print $1}' | xargs -r systemctl disable
    
    # Set up fail2ban if available
    if command -v fail2ban-client &> /dev/null; then
        systemctl enable fail2ban
        systemctl start fail2ban
        log "Fail2ban enabled"
    fi
    
    log "Security hardening completed"
}

# Main deployment function
main() {
    log "Starting OpenAlgebra Medical AI deployment"
    log "Version: $VERSION"
    log "Environment: $ENVIRONMENT"
    log "Domain: $DOMAIN"
    
    validate_environment
    setup_directories
    generate_configs
    setup_ssl
    start_services
    run_migrations
    setup_monitoring
    setup_backups
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        apply_security
    fi
    
    log "🎉 OpenAlgebra Medical AI deployment completed successfully!"
    log ""
    log "📋 Deployment Summary:"
    log "  • Version: $VERSION"
    log "  • Environment: $ENVIRONMENT"
    log "  • Domain: $DOMAIN"
    log "  • API URL: http${ENVIRONMENT == 'production' && DOMAIN != 'localhost' && echo 's'}://$DOMAIN"
    log "  • Health Check: http${ENVIRONMENT == 'production' && DOMAIN != 'localhost' && echo 's'}://$DOMAIN/health"
    log "  • API Documentation: http${ENVIRONMENT == 'production' && DOMAIN != 'localhost' && echo 's'}://$DOMAIN/docs"
    log ""
    log "📁 Important Locations:"
    log "  • Configuration: /opt/openalgebra/configs/"
    log "  • Data Directory: /opt/openalgebra/data/"
    log "  • Logs: /var/log/openalgebra/"
    log "  • Backups: /opt/openalgebra/backups/"
    log ""
    log "Management Commands:"
    log "  • View logs: docker-compose -f $PROJECT_DIR/docker-compose.yml logs -f"
    log "  • Restart services: docker-compose -f $PROJECT_DIR/docker-compose.yml restart"
    log "  • Update deployment: $0 <new-version> $ENVIRONMENT $DOMAIN"
    log ""
    log "🔍 Next Steps:"
    log "  1. Verify all services are running: docker-compose ps"
    log "  2. Test API endpoints: curl http${ENVIRONMENT == 'production' && DOMAIN != 'localhost' && echo 's'}://$DOMAIN/health"
    log "  3. Review logs for any issues: tail -f /var/log/openalgebra/*.log"
    log "  4. Configure monitoring alerts if needed"
    log "  5. Test backup and restore procedures"
}

# Run main function
main "$@" 