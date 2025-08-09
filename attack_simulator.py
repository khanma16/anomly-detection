"""
Network Attack Simulator for Anomaly Detection Testing

COMPREHENSIVE USAGE GUIDE:
=========================

BASIC COMMANDS:
# Linux/Mac (requires sudo for packet injection)
sudo venv/bin/python attack_simulator.py --safe
sudo venv/bin/python attack_simulator.py --attack port_scan --duration 30

# Windows (Run Command Prompt as Administrator)
venv\Scripts\python.exe attack_simulator.py --safe
venv\Scripts\python.exe attack_simulator.py --attack dos --duration 20

ATTACK TYPES AVAILABLE:
======================
--attack port_scan      # TCP SYN packets to common ports (21,22,80,443,etc)
--attack dos            # High-volume SYN/UDP/ICMP flood attacks
--attack suspicious     # Connections to unusual ports (1000-65535)
--attack large_packets  # Oversized payloads (1400-9000 bytes)
--attack icmp_flood     # ICMP burst attacks (various types)
--attack all           # Sequential execution of all attack types (DEFAULT)

INTERFACE & TARGET OPTIONS:
==========================
--target 127.0.0.1     # Loopback (SAFE - recommended for demos)
--target 192.168.1.10  # LAN target (requires network permission)
--interface lo          # Loopback interface (safe)
--interface eth0        # Ethernet interface (real network)
--interface wlan0       # WiFi interface (wireless network)
--safe                  # Force safe mode (127.0.0.1 + lo interface)

DURATION & TIMING:
=================
--duration 10          # Attack duration in seconds (default: 30)
# Suggested durations by attack type:
# port_scan: 20-60s    (slower, methodical scanning)
# dos: 10-30s          (high intensity, short bursts)
# suspicious: 15-45s   (moderate pace connection attempts)
# large_packets: 5-20s (slow, large payload transmission)
# icmp_flood: 5-15s    (rapid packet bursts)

APP INTEGRATION WITH ANOMALY DETECTION:
======================================
1. START ANOMALY DETECTION APP:
   # Linux/Mac
   sudo venv/bin/python app.py
   # Windows
   venv\Scripts\python.exe app.py
   # Open: http://localhost:8000

2. CONFIGURE DETECTION SETTINGS:
   Interface Options:
   - lo (Loopback - Safe)        # Use with --safe attacks
   - eth0 (Ethernet)             # Use with LAN attacks
   - wlan0 (WiFi)                # Use with wireless attacks

   Dataset Model Options:
   - NSL-KDD Models              # Best for: port_scan, dos (85-95% detection)
   - CICIDS2017 Models           # Best for: large_packets, dos (90-98% detection)
   - UNSW-NB15 Models            # Best for: suspicious, all attacks (85-95% detection)
   - TON-IoT Models              # Best for: icmp_flood, IoT attacks (60-80% detection)

3. RUN COORDINATED ATTACKS:
   # Safe demo setup (recommended)
   sudo venv/bin/python attack_simulator.py --safe --attack all

   # Specific dataset testing
   sudo venv/bin/python attack_simulator.py --safe --attack port_scan --duration 30  # NSL-KDD
   sudo venv/bin/python attack_simulator.py --safe --attack large_packets --duration 15  # CICIDS2017
   sudo venv/bin/python attack_simulator.py --safe --attack suspicious --duration 25  # UNSW-NB15
   sudo venv/bin/python attack_simulator.py --safe --attack icmp_flood --duration 10  # TON-IoT

DEMO SCENARIOS:
==============
# Demo
sudo venv/bin/python attack_simulator.py --safe --attack all

# Research Testing (specific models)
sudo venv/bin/python attack_simulator.py --safe --attack port_scan --duration 60
sudo venv/bin/python attack_simulator.py --safe --attack dos --duration 30
sudo venv/bin/python attack_simulator.py --safe --attack suspicious --duration 45

# Performance Benchmarking
sudo venv/bin/python attack_simulator.py --safe --attack dos --duration 120
sudo venv/bin/python attack_simulator.py --safe --attack large_packets --duration 60

WARNING: Use only on networks you own or have explicit permission!
"""

import time
import random
import argparse
import signal
import sys
import platform

try:
    from scapy.all import IP, TCP, UDP, ICMP, Raw, send

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("ERROR: Scapy not available. Install with: pip install scapy")


class NetworkAttackSimulator:
    def __init__(self, target_ip="127.0.0.1", interface="lo"):
        self.target_ip = target_ip
        self.interface = interface
        self.is_running = True
        self.stats = {
            "port_scan": 0,
            "dos_packets": 0,
            "suspicious_connections": 0,
            "large_packets": 0,
            "icmp_packets": 0,
        }

        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy is required for packet generation")
            return

    def port_scan_attack(self, duration=30):
        print(f"Starting port scan attack on {self.target_ip} for {duration}s...")

        if not SCAPY_AVAILABLE:
            return 0

        start_time = time.time()
        ports_scanned = 0
        common_ports = [
            21,
            22,
            23,
            25,
            53,
            80,
            110,
            143,
            443,
            993,
            995,
            8080,
            8443,
            3389,
            5432,
            3306,
            1433,
            6379,
            27017,
        ]

        try:
            while time.time() - start_time < duration and self.is_running:
                port = random.choice(common_ports)
                packet = IP(dst=self.target_ip) / TCP(dport=port, flags="S")

                try:
                    send(packet, verbose=0)
                    ports_scanned += 1
                    self.stats["port_scan"] += 1
                except Exception as e:
                    if ports_scanned == 0:  # Only show first error
                        print(f"Port scan error: {e}")

                time.sleep(random.uniform(0.1, 0.5))

        except KeyboardInterrupt:
            pass

        print(f"Port scan completed: {ports_scanned} ports scanned")
        return ports_scanned

    def dos_attack(self, duration=15):
        print(f"Starting DoS attack on {self.target_ip} for {duration}s...")

        if not SCAPY_AVAILABLE:
            return 0

        start_time = time.time()
        packets_sent = 0

        try:
            while time.time() - start_time < duration and self.is_running:
                attack_types = [
                    IP(dst=self.target_ip) / TCP(dport=80, flags="S"),
                    IP(dst=self.target_ip) / UDP(dport=53),
                    IP(dst=self.target_ip) / ICMP(),
                ]

                packet = random.choice(attack_types)

                try:
                    for _ in range(random.randint(3, 10)):
                        send(packet, verbose=0)
                        packets_sent += 1
                        self.stats["dos_packets"] += 1

                        if not self.is_running:
                            break

                except Exception as e:
                    if packets_sent == 0:  # Only show first error
                        print(f"DoS attack error: {e}")

                time.sleep(random.uniform(0.01, 0.1))

        except KeyboardInterrupt:
            pass

        print(f"DoS attack completed: {packets_sent} packets sent")
        return packets_sent

    def suspicious_connections(self, duration=15):
        print(f"Starting suspicious connections to {self.target_ip} for {duration}s...")

        if not SCAPY_AVAILABLE:
            return 0

        start_time = time.time()
        connections = 0

        try:
            while time.time() - start_time < duration and self.is_running:
                patterns = [
                    lambda: [
                        IP(dst=self.target_ip) / TCP(dport=p, flags="S")
                        for p in random.sample(range(1000, 9999), 5)
                    ],
                    lambda: [
                        IP(dst=self.target_ip)
                        / TCP(dport=random.randint(30000, 65535), flags="S")
                    ],
                    lambda: [
                        IP(dst=self.target_ip) / TCP(dport=22, flags="S")
                        for _ in range(10)
                    ],
                ]

                pattern = random.choice(patterns)
                packets = pattern()

                try:
                    for packet in packets:
                        send(packet, verbose=0)
                        connections += 1
                        self.stats["suspicious_connections"] += 1

                        if not self.is_running:
                            break
                except Exception as e:
                    if connections == 0:  # Only show first error
                        print(f"Suspicious connections error: {e}")

                time.sleep(random.uniform(0.2, 1.0))

        except KeyboardInterrupt:
            pass

        print(f"Suspicious connections completed: {connections} attempts")
        return connections

    def large_packet_attack(self, duration=10):
        print(f"Starting large packet attack on {self.target_ip} for {duration}s...")

        if not SCAPY_AVAILABLE:
            return 0

        start_time = time.time()
        packets_sent = 0

        try:
            while time.time() - start_time < duration and self.is_running:
                payload_sizes = [1400, 2000, 4000, 8000, 9000]
                payload_size = random.choice(payload_sizes)
                payload = "A" * payload_size

                packet_types = [
                    IP(dst=self.target_ip) / TCP(dport=80) / Raw(load=payload),
                    IP(dst=self.target_ip) / UDP(dport=53) / Raw(load=payload),
                    IP(dst=self.target_ip) / ICMP() / Raw(load=payload),
                ]

                packet = random.choice(packet_types)

                try:
                    send(packet, verbose=0)
                    packets_sent += 1
                    self.stats["large_packets"] += 1
                except Exception as e:
                    if packets_sent == 0:  # Only show first error
                        print(f"Large packet error: {e}")

                time.sleep(random.uniform(0.5, 2.0))

        except KeyboardInterrupt:
            pass

        print(f"Large packet attack completed: {packets_sent} large packets sent")
        return packets_sent

    def icmp_flood(self, duration=10):
        print(f"Starting ICMP flood on {self.target_ip} for {duration}s...")

        if not SCAPY_AVAILABLE:
            return 0

        start_time = time.time()
        packets_sent = 0

        try:
            while time.time() - start_time < duration and self.is_running:
                icmp_types = [8, 13, 15, 17]
                icmp_type = random.choice(icmp_types)
                packet = IP(dst=self.target_ip) / ICMP(type=icmp_type)

                try:
                    for _ in range(random.randint(5, 15)):
                        send(packet, verbose=0)
                        packets_sent += 1
                        self.stats["icmp_packets"] += 1

                        if not self.is_running:
                            break

                except Exception as e:
                    if packets_sent == 0:  # Only show first error
                        print(f"ICMP flood error: {e}")

                time.sleep(random.uniform(0.1, 0.5))

        except KeyboardInterrupt:
            pass

        print(f"ICMP flood completed: {packets_sent} packets sent")
        return packets_sent

    def run_all_attacks(self):
        if not SCAPY_AVAILABLE:
            return

        attacks = [
            ("Port Scan", self.port_scan_attack, 30),
            ("Suspicious Connections", self.suspicious_connections, 15),
            ("Large Packets", self.large_packet_attack, 10),
            ("ICMP Flood", self.icmp_flood, 10),
            ("DoS Simulation", self.dos_attack, 15),
        ]

        print(f"Starting {len(attacks)} attack simulations...")
        print("Monitor your anomaly detection system now!")
        print("Press Ctrl+C to stop")

        try:
            for attack_name, attack_func, duration in attacks:
                if not self.is_running:
                    break

                print(f"\n--- {attack_name} ---")
                attack_func(duration)

                if not self.is_running:
                    print("Attack simulation stopped by user")
                    break

                print("Pausing 3 seconds before next attack...")
                time.sleep(3)

        except KeyboardInterrupt:
            print("Attack simulation stopped by user")

        self.is_running = False
        print("\nAll attack simulations completed!")
        self.print_statistics()

    def run_specific_attack(self, attack_type, duration=30):
        if not SCAPY_AVAILABLE:
            return

        attack_map = {
            "port_scan": self.port_scan_attack,
            "dos": self.dos_attack,
            "suspicious": self.suspicious_connections,
            "large_packets": self.large_packet_attack,
            "icmp_flood": self.icmp_flood,
        }

        if attack_type not in attack_map:
            print(f"ERROR: Unknown attack type: {attack_type}")
            return

        try:
            print(f"Starting {attack_type} attack on {self.target_ip}")
            attack_map[attack_type](duration)

        except KeyboardInterrupt:
            print("Attack stopped by user")

        self.print_statistics()

    def print_statistics(self):
        print("\nAttack Statistics:")
        print("-" * 30)
        for attack_type, count in self.stats.items():
            print(f"{attack_type.replace('_', ' ').title()}: {count}")

    def stop(self):
        self.is_running = False


def signal_handler(sig, frame):
    global simulator
    print("\nStopping attack simulation...")
    simulator.stop()
    sys.exit(0)


def show_platform_info():
    os_name = platform.system()
    print(f"Platform: {os_name}")

    if os_name == "Windows":
        print("Windows: Run Command Prompt as Administrator")
        print("Command: venv\\Scripts\\python.exe attack_simulator.py --safe")
    else:
        print("Linux/Mac: Use sudo for packet sending privileges")
        print("Command: sudo venv/bin/python attack_simulator.py --safe")


def main():
    if not SCAPY_AVAILABLE:
        print("ERROR: Scapy is required for attack simulation")
        print("Install with: pip install scapy")
        return

    parser = argparse.ArgumentParser(
        description="Cross-Platform Network Attack Simulator"
    )
    parser.add_argument(
        "--target", default="127.0.0.1", help="Target IP address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--interface", default="lo", help="Network interface (default: lo)"
    )
    parser.add_argument(
        "--attack",
        choices=[
            "port_scan",
            "dos",
            "suspicious",
            "large_packets",
            "icmp_flood",
            "all",
        ],
        default="all",
        help="Attack type to run",
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Attack duration in seconds"
    )
    parser.add_argument(
        "--safe", action="store_true", help="Force safe mode (127.0.0.1 target)"
    )
    parser.add_argument(
        "--info", action="store_true", help="Show platform-specific usage information"
    )

    args = parser.parse_args()

    if args.info:
        show_platform_info()
        return

    if args.target != "127.0.0.1" and not args.safe:
        confirm = input(
            f"WARNING: Target is {args.target}. Are you authorized to test this target? (yes/no): "
        )
        if confirm.lower() != "yes":
            print("ERROR: Attack cancelled for safety")
            return

    if args.safe:
        args.target = "127.0.0.1"
        args.interface = "lo"

    global simulator
    simulator = NetworkAttackSimulator(args.target, args.interface)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.attack == "all":
            simulator.run_all_attacks()
        else:
            simulator.run_specific_attack(args.attack, args.duration)

    except KeyboardInterrupt:
        print("Attack simulation stopped")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
