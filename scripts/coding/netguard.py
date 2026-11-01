"""Only this trusted setup process receives NET_ADMIN, in its own Docker namespace."""

import ipaddress, os, subprocess, sys

address, port, subnet, seconds = sys.argv[1:]
ipaddress.IPv4Address(address)
ipaddress.IPv4Network(subnet)
assert 1 <= int(port) <= 65535 and 1 <= int(seconds) <= 180
for binary in ("/usr/sbin/iptables", "/usr/sbin/ip6tables"):
    for chain in ("INPUT", "OUTPUT", "FORWARD"):
        subprocess.run([binary, "-w", "2", "-P", chain, "DROP"], check=True)
    for chain in ("INPUT", "OUTPUT"):
        subprocess.run(
            [
                binary,
                "-w",
                "2",
                "-A",
                chain,
                "-m",
                "conntrack",
                "--ctstate",
                "ESTABLISHED,RELATED",
                "-j",
                "ACCEPT",
            ],
            check=True,
        )
subprocess.run(
    [
        "/usr/sbin/iptables",
        "-w",
        "2",
        "-A",
        "INPUT",
        "-s",
        subnet,
        "-p",
        "tcp",
        "--dport",
        "3030",
        "-j",
        "ACCEPT",
    ],
    check=True,
)
subprocess.run(
    [
        "/usr/sbin/iptables",
        "-w",
        "2",
        "-A",
        "OUTPUT",
        "-d",
        address,
        "-p",
        "tcp",
        "--dport",
        port,
        "-j",
        "ACCEPT",
    ],
    check=True,
)
os.execvp("timeout", ["timeout", "--signal=KILL", seconds, "sleep", "181"])
