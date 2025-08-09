"""
Email Alert System for Anomaly Detection
=======================================

Handles sending notification emails when anomalies are detected.
Supports various SMTP servers and provides templated email messages.

Usage:
    # Linux/Mac
    source venv/bin/activate
    python src/alert_system.py
    
    # Windows
    venv\Scripts\activate
    python src/alert_system.py

"""

import smtplib
import yaml
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any
import os
import json
import time
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)

class EmailAlerter:
    """Email alerting system for anomaly detection notifications."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the email alerter with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.email_config = self.config.get('email', {})
        self.last_alert_time = 0
        self.alert_count = 0
        self.rate_limit_lock = Lock()
        
        # Rate limiting settings to prevent spam
        self.min_alert_interval = 60  # Minimum seconds between alerts
        self.max_alerts_per_hour = 10
        self.hourly_alert_count = 0
        self.hour_start_time = time.time()
        
        logger.info("Email alerter initialized")
    
    def _check_rate_limit(self) -> bool:
        """Check if we should send an alert based on rate limiting rules."""
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Reset hourly counter if an hour has passed
            if current_time - self.hour_start_time >= 3600:
                self.hourly_alert_count = 0
                self.hour_start_time = current_time
            
            # Check minimum interval between alerts
            if current_time - self.last_alert_time < self.min_alert_interval:
                logger.info(f"Alert rate limited: {self.min_alert_interval}s interval not met")
                return False
            
            # Check maximum alerts per hour
            if self.hourly_alert_count >= self.max_alerts_per_hour:
                logger.info(f"Alert rate limited: {self.max_alerts_per_hour} alerts/hour exceeded")
                return False
            
            # Update counters
            self.last_alert_time = current_time
            self.hourly_alert_count += 1
            self.alert_count += 1
            
            return True
    
    def _create_smtp_connection(self) -> smtplib.SMTP:
        """Create and configure SMTP connection."""
        try:
            smtp_server = smtplib.SMTP(
                self.email_config['smtp_server'], 
                self.email_config['smtp_port']
            )
            
            # Enable TLS encryption
            smtp_server.starttls()
            
            # Login to the server
            smtp_server.login(
                self.email_config['sender_email'], 
                self.email_config['sender_password']
            )
            
            logger.debug("SMTP connection established successfully")
            return smtp_server
            
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {str(e)}")
            raise
    
    def _format_anomaly_details(self, alert_data: Dict[str, Any]) -> str:
        """Format anomaly details for email body."""
        details = []
        
        # Basic information
        details.append(f"Timestamp: {alert_data['timestamp']}")
        details.append(f"Row Index: {alert_data['row_index']}")
        details.append(f"Best Model: {alert_data['model_name']}")
        details.append(f"Confidence: {alert_data['confidence']:.4f}")
        details.append("")
        
        # Model predictions
        details.append("Model Predictions:")
        details.append("-" * 20)
        
        for model_name, prediction in alert_data.get('predictions', {}).items():
            details.append(f"{model_name.upper()}:")
            for key, value in prediction.items():
                if isinstance(value, float):
                    details.append(f"  {key}: {value:.4f}")
                else:
                    details.append(f"  {key}: {value}")
            details.append("")
        
        # Feature summary showing top 10 most significant features
        features = alert_data.get('features', {})
        if features:
            details.append("Key Features (Top 10):")
            details.append("-" * 20)
            
            # Sort features by absolute value to show most significant
            sorted_features = sorted(
                features.items(), 
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )[:10]
            
            for feature, value in sorted_features:
                if isinstance(value, float):
                    details.append(f"  {feature}: {value:.4f}")
                else:
                    details.append(f"  {feature}: {value}")
        
        return "\n".join(details)
    
    def _create_email_message(self, alert_data: Dict[str, Any]) -> MIMEMultipart:
        """Create email message with anomaly alert details."""
        msg = MIMEMultipart('alternative')
        
        # Set headers
        msg['From'] = self.email_config['sender_email']
        msg['To'] = self.email_config['recipient_email']
        msg['Subject'] = self.email_config['subject_template']
        
        # Format email body using template
        body_template = self.email_config['body_template']
        
        # Prepare template variables
        template_vars = {
            'timestamp': alert_data['timestamp'],
            'score': alert_data['confidence'],
            'model_name': alert_data['model_name'],
            'features': self._format_anomaly_details(alert_data)
        }
        
        # Format the body with fallback for missing template variables
        try:
            email_body = body_template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template variable missing: {e}. Using fallback template.")
            email_body = f"""
Network Anomaly Alert

Timestamp: {alert_data['timestamp']}
Model: {alert_data['model_name']}
Confidence: {alert_data['confidence']:.4f}

{self._format_anomaly_details(alert_data)}

Please investigate this anomaly immediately.
"""
        
        # Create plain text part
        text_part = MIMEText(email_body, 'plain')
        msg.attach(text_part)
        
        # Create HTML version for better formatting
        html_body = email_body.replace('\n', '<br>')
        html_body = f"""
        <html>
        <body>
        <div style="font-family: Arial, sans-serif; max-width: 600px;">
        <h2 style="color: #d73027;">NETWORK ANOMALY ALERT</h2>
        <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #d73027;">
        <pre style="white-space: pre-wrap; font-family: monospace;">{html_body}</pre>
        </div>
        <p style="color: #666; font-size: 12px;">
        This is an automated alert from the Network Anomaly Detection System.
        </p>
        </div>
        </body>
        </html>
        """
        
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        return msg
    
    def send_anomaly_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send anomaly alert email with rate limiting."""
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                return False
            
            logger.info(f"Sending anomaly alert for row {alert_data.get('row_index', 'unknown')}")
            
            # Create and send email message
            message = self._create_email_message(alert_data)
            
            smtp_server = None
            try:
                smtp_server = self._create_smtp_connection()
                smtp_server.send_message(message)
                logger.info(f"Anomaly alert sent successfully (Alert #{self.alert_count})")
            finally:
                if smtp_server:
                    try:
                        smtp_server.quit()
                    except:
                        pass  # Ignore errors when closing connection
            
            # Log alert details for audit trail
            self._log_alert(alert_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send anomaly alert: {str(e)}")
            return False
    
    def _log_alert(self, alert_data: Dict[str, Any]) -> None:
        """Log alert details to a separate file for audit purposes."""
        try:
            # Create alerts log directory
            log_dir = os.path.dirname(self.config['logging']['log_file'])
            alerts_log_path = os.path.join(log_dir, 'email_alerts.log')
            
            # Prepare log entry
            log_entry = {
                'alert_id': self.alert_count,
                'timestamp': datetime.now().isoformat(),
                'alert_data': alert_data,
                'recipient': self.email_config['recipient_email']
            }
            
            # Write to log file
            with open(alerts_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log alert details: {str(e)}")
    
    def send_test_email(self) -> bool:
        """Send a test email to verify configuration."""
        try:
            print("Sending test email...")
            logger.info("Sending test email...")
            
            # Create simple test message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = "TEST: Anomaly Detection System"
            
            body = f"""
This is a test email from your Network Traffic Anomaly Detection System.

If you receive this email, the email alert system is working correctly.

Test Details:
- Timestamp: {datetime.now().isoformat()}
- System: Anomaly Detection
- Status: Testing Email Functionality

Best regards,
Network Traffic Anomaly Detection System
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email with timeout handling
            print("Connecting to SMTP server...")
            server = smtplib.SMTP(
                self.email_config['smtp_server'], 
                self.email_config['smtp_port'],
                timeout=30  # 30 second timeout
            )
            
            print("Starting TLS encryption...")
            server.starttls()
            
            print("Logging in...")
            server.login(
                self.email_config['sender_email'], 
                self.email_config['sender_password']
            )
            
            print("Sending email...")
            server.send_message(msg)
            
            print("Closing connection...")
            server.quit()
            
            print("SUCCESS: Test email sent successfully!")
            logger.info("Test email sent successfully")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP Authentication failed: {e}"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            return False
            
        except smtplib.SMTPConnectError as e:
            error_msg = f"Could not connect to SMTP server: {e}"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            return False
            
        except Exception as e:
            error_msg = f"Error sending test email: {str(e)}"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            return False
    
    def test_connection(self) -> bool:
        """Test SMTP connection without sending email."""
        try:
            print("Testing SMTP connection...")
            server = self._create_smtp_connection()
            server.quit()
            print("SUCCESS: SMTP connection test passed")
            return True
        except Exception as e:
            print(f"ERROR: SMTP connection test failed: {e}")
            return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get email alert statistics."""
        return {
            'total_alerts_sent': self.alert_count,
            'hourly_alerts': self.hourly_alert_count,
            'last_alert_time': datetime.fromtimestamp(self.last_alert_time).isoformat() if self.last_alert_time > 0 else None,
            'rate_limit_active': time.time() - self.last_alert_time < self.min_alert_interval,
            'max_alerts_per_hour': self.max_alerts_per_hour,
            'min_alert_interval': self.min_alert_interval
        }


def main():
    """Main function - sends test email to verify configuration."""
    try:
        print("Network Anomaly Detection - Email Alert System Test")
        print("=" * 55)
        
        alerter = EmailAlerter()
        alerter.send_test_email()
        
    except Exception as e:
        logger.error(f"Error in email alerter test: {str(e)}")
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main() 