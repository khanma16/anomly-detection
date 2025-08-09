#!/usr/bin/env python3
"""
Network Attack Simulator for Anomaly Detection Testing
=====================================================

CRITICAL WARNING: Use only on networks you own or have explicit permission!
Unauthorized network attacks are illegal and unethical.

This script generates various network attacks for testing anomaly detection systems:
- Port scanning
- DoS attacks  
- Suspicious connections
- Large packet attacks
- ICMP flooding

Usage:
    python attack_simulator.py --safe                    # Safe loopback testing
    python attack_simulator.py --target 192.168.1.10   # Specific target
    python attack_simulator.py --attack dos --duration 30  # Specific attack
"""

import time
import random
import socket
import threading
import argparse
import signal
import sys
from datetime import datetime

try:
    from scapy.all import IP, TCP, UDP, ICMP, Raw, send, srp1, ARP, Ether
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("WARNING: Scapy not available. Install with: pip install scapy")

class NetworkAttackSimulator:
    def __init__(self, target_ip="127.0.0.1", interface="lo"):
        if not SCAPY_AVAILABLE:
            print("NETWORK ATTACK SIMULATOR")
            print("=" * 40)
            print("WARNING: Use only on networks you own!")
            print("WARNING: Unauthorized attacks are illegal!")
            print("WARNING: This is for testing anomaly detection only!")
            print("=" * 40)
            return
            
        self.target_ip = target_ip
        self.interface = interface
        self.is_running = True
        
        # Statistics tracking
        self.stats = {
            'port_scan': 0,
            'dos_packets': 0,
            'suspicious_connections': 0,
            'large_packets': 0,
            'icmp_packets': 0
        }
    
    def port_scan_attack(self, duration=30):
        """Simulate port scanning attack"""
        print(f"Starting port scan attack on {self.target_ip} for {duration}s...")
        
        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy required for port scan attack")
            return
        
        start_time = time.time()
        ports_scanned = 0
        
        # Common ports to scan
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 
                       8080, 8443, 3389, 5432, 3306, 1433, 6379, 27017]
        
        try:
            while time.time() - start_time < duration and self.is_running:
                port = random.choice(common_ports)
                
                # Create TCP SYN packet for port scan
                packet = IP(dst=self.target_ip) / TCP(dport=port, flags="S")
                
                try:
                    send(packet, verbose=0, timeout=1)
                    ports_scanned += 1
                    self.stats['port_scan'] += 1
                except:
                    pass  # Ignore send errors
                
                # Random delay between scans
                time.sleep(random.uniform(0.1, 0.5))
                
        except KeyboardInterrupt:
            pass
        
        print(f"Port scan completed: {ports_scanned} ports scanned")
        return ports_scanned
    
    def dos_attack(self, duration=15):
        """Simulate DoS attack with high packet rate"""
        print(f"Starting DoS attack on {self.target_ip} for {duration}s...")
        
        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy required for DoS attack")
            return
        
        start_time = time.time()
        packets_sent = 0
        
        try:
            while time.time() - start_time < duration and self.is_running:
                # Create multiple packet types for DoS
                attack_types = [
                    IP(dst=self.target_ip) / TCP(dport=80, flags="S"),  # SYN flood
                    IP(dst=self.target_ip) / UDP(dport=53),  # UDP flood
                    IP(dst=self.target_ip) / ICMP(),  # ICMP flood
                ]
                
                packet = random.choice(attack_types)
                
                try:
                    # Send multiple packets rapidly
                    for _ in range(random.randint(3, 10)):
                        send(packet, verbose=0, timeout=0.1)
                        packets_sent += 1
                        self.stats['dos_packets'] += 1
                        
                        if not self.is_running:
                            break
                            
                except:
                    pass  # Ignore send errors
                
                # Brief pause to prevent overwhelming
                time.sleep(random.uniform(0.01, 0.1))
                
        except KeyboardInterrupt:
            pass
        
        print(f"DoS attack completed: {packets_sent} packets sent")
        return packets_sent
    
    def suspicious_connections(self, duration=15):
        """Generate suspicious connection patterns"""
        print(f"Starting suspicious connections to {self.target_ip} for {duration}s...")
        
        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy required for suspicious connections")
            return
        
        start_time = time.time()
        connections = 0
        
        try:
            while time.time() - start_time < duration and self.is_running:
                # Suspicious patterns
                patterns = [
                    # Multiple connections from same source
                    lambda: [IP(dst=self.target_ip) / TCP(dport=p, flags="S") 
                            for p in random.sample(range(1000, 9999), 5)],
                    # Connections to uncommon ports
                    lambda: [IP(dst=self.target_ip) / TCP(dport=random.randint(30000, 65535), flags="S")],
                    # Rapid connection attempts
                    lambda: [IP(dst=self.target_ip) / TCP(dport=22, flags="S") for _ in range(10)]
                ]
                
                pattern = random.choice(patterns)
                packets = pattern()
                
                try:
                    for packet in packets:
                        send(packet, verbose=0, timeout=0.1)
                        connections += 1
                        self.stats['suspicious_connections'] += 1
                        
                        if not self.is_running:
                            break
                except:
                    pass
                
                time.sleep(random.uniform(0.2, 1.0))
                
        except KeyboardInterrupt:
            pass
        
        print(f"Suspicious connections completed: {connections} attempts")
        return connections
    
    def large_packet_attack(self, duration=10):
        """Send unusually large packets"""
        print(f"Starting large packet attack on {self.target_ip} for {duration}s...")
        
        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy required for large packet attack")
            return
        
        start_time = time.time()
        packets_sent = 0
        
        try:
            while time.time() - start_time < duration and self.is_running:
                # Create large payloads (potential buffer overflow attempts)
                payload_sizes = [1400, 2000, 4000, 8000, 9000]  # Various large sizes
                payload_size = random.choice(payload_sizes)
                payload = "A" * payload_size
                
                # Different protocols with large payloads
                packet_types = [
                    IP(dst=self.target_ip) / TCP(dport=80) / Raw(load=payload),
                    IP(dst=self.target_ip) / UDP(dport=53) / Raw(load=payload),
                    IP(dst=self.target_ip) / ICMP() / Raw(load=payload)
                ]
                
                packet = random.choice(packet_types)
                
                try:
                    send(packet, verbose=0, timeout=1)
                    packets_sent += 1
                    self.stats['large_packets'] += 1
                except:
                    pass
                
                time.sleep(random.uniform(0.5, 2.0))
                
        except KeyboardInterrupt:
            pass
        
        print(f"Large packet attack completed: {packets_sent} large packets sent")
        return packets_sent
    
    def icmp_flood(self, duration=10):
        """ICMP flood attack"""
        print(f"Starting ICMP flood on {self.target_ip} for {duration}s...")
        
        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy required for ICMP flood")
            return
        
        start_time = time.time()
        packets_sent = 0
        
        try:
            while time.time() - start_time < duration and self.is_running:
                # Various ICMP packet types
                icmp_types = [8, 13, 15, 17]  # Echo, Timestamp, Info Request, Address Mask
                icmp_type = random.choice(icmp_types)
                
                packet = IP(dst=self.target_ip) / ICMP(type=icmp_type)
                
                try:
                    # Send bursts of ICMP packets
                    for _ in range(random.randint(5, 15)):
                        send(packet, verbose=0, timeout=0.1)
                        packets_sent += 1
                        self.stats['icmp_packets'] += 1
                        
                        if not self.is_running:
                            break
                            
                except:
                    pass
                
                time.sleep(random.uniform(0.1, 0.5))
                
        except KeyboardInterrupt:
            pass
        
        print(f"ICMP flood completed: {packets_sent} packets sent")
        return packets_sent
    
    def run_all_attacks(self):
        """Run all attack types in sequence"""
        if not SCAPY_AVAILABLE:
            return
        
        attacks = [
            ("Port Scan", self.port_scan_attack, 30),
            ("Suspicious Connections", self.suspicious_connections, 15),
            ("Large Packets", self.large_packet_attack, 10),
            ("ICMP Flood", self.icmp_flood, 10),
            ("DoS Simulation", self.dos_attack, 15)
        ]
        
        print(f"\nStarting {len(attacks)} attack simulations...")
        print("Monitor your anomaly detection system now!")
        print("Press Ctrl+C to stop\n")
        
        try:
            for attack_name, attack_func, duration in attacks:
                if not self.is_running:
                    break
                
                print(f"--- {attack_name} ---")
                attack_func(duration)
                
                if not self.is_running:
                    print("\nAttack simulation stopped by user")
                    break
                    
                print("Pausing 3 seconds before next attack...")
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\nAttack simulation stopped by user")
        except Exception as e:
            print(f"ERROR: Error in {attack_name}: {e}")
            
        self.is_running = False
        
        print("\nAll attack simulations completed!")
        self.print_statistics()
    
    def run_specific_attack(self, attack_type, duration=30):
        """Run a specific attack type"""
        if not SCAPY_AVAILABLE:
            return
        
        attack_map = {
            'port_scan': self.port_scan_attack,
            'dos': self.dos_attack,
            'suspicious': self.suspicious_connections,
            'large_packets': self.large_packet_attack,
            'icmp_flood': self.icmp_flood
        }
        
        if attack_type not in attack_map:
            print(f"ERROR: Unknown attack type: {attack_type}")
            return
        
        try:
            print(f"Starting {attack_type} attack on {self.target_ip}")
            attack_map[attack_type](duration)
            
        except KeyboardInterrupt:
            print("\nAttack stopped by user")
        except Exception as e:
            print(f"ERROR: Attack error: {e}")
        
        self.print_statistics()
    
    def generate_normal_traffic(self, duration=60):
        """Generate normal-looking traffic for baseline"""
        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy is required for traffic generation")
            return
        
        start_time = time.time()
        packets_sent = 0
        
        print(f"Generating normal traffic to {self.target_ip} for {duration}s...")
        
        try:
            while time.time() - start_time < duration and self.is_running:
                # Normal web traffic
                normal_packets = [
                    IP(dst=self.target_ip) / TCP(dport=80, flags="S"),  # HTTP
                    IP(dst=self.target_ip) / TCP(dport=443, flags="S"),  # HTTPS
                    IP(dst=self.target_ip) / UDP(dport=53),  # DNS
                    IP(dst=self.target_ip) / TCP(dport=22, flags="S"),  # SSH
                ]
                
                packet = random.choice(normal_packets)
                
                try:
                    send(packet, verbose=0, timeout=1)
                    packets_sent += 1
                except:
                    pass
                
                # Normal intervals between packets
                time.sleep(random.uniform(1.0, 5.0))
                
        except KeyboardInterrupt:
            pass
        
        print(f"Normal traffic generation completed: {packets_sent} packets")
        return packets_sent
    
    def print_statistics(self):
        """Print attack statistics"""
        print("\nAttack Statistics:")
        print("-" * 30)
        for attack_type, count in self.stats.items():
            print(f"{attack_type.replace('_', ' ').title()}: {count}")
    
    def stop(self):
        """Stop all attacks"""
        self.is_running = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global simulator
    print('\nStopping attack simulation...')
    simulator.stop()
    sys.exit(0)

def main():
    if not SCAPY_AVAILABLE:
        print("ERROR: Scapy is required for attack simulation")
        print("Install with: pip install scapy")
        return
    
    parser = argparse.ArgumentParser(description='Network Attack Simulator for Anomaly Detection Testing')
    parser.add_argument('--target', default='127.0.0.1', help='Target IP address (default: 127.0.0.1)')
    parser.add_argument('--interface', default='lo', help='Network interface (default: lo)')
    parser.add_argument('--attack', choices=['port_scan', 'dos', 'suspicious', 'large_packets', 'icmp_flood', 'all'], 
                       default='all', help='Attack type to run')
    parser.add_argument('--duration', type=int, default=30, help='Attack duration in seconds')
    parser.add_argument('--safe', action='store_true', help='Force safe mode (127.0.0.1 target)')
    
    args = parser.parse_args()
    
    # Safety check for non-loopback targets
    if args.target != '127.0.0.1' and not args.safe:
        confirm = input(f"WARNING: Target is {args.target}. Are you authorized to test this target? (yes/no): ")
        if confirm.lower() != 'yes':
            print("ERROR: Attack cancelled for safety")
            return
    
    # Force safe mode if requested
    if args.safe:
        args.target = '127.0.0.1'
        args.interface = 'lo'
    
    global simulator
    simulator = NetworkAttackSimulator(args.target, args.interface)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.attack == 'all':
            simulator.run_all_attacks()
        else:
            simulator.run_specific_attack(args.attack, args.duration)
            
    except KeyboardInterrupt:
        print("\nAttack simulation stopped")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main() 