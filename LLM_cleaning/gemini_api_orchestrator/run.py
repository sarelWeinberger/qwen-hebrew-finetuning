#!/usr/bin/env python3
"""
Gepeta EC2 Run - נקודת כניסה ראשית
"""

import subprocess
import sys


def main():
    print("🚀 Gepeta EC2 Orchestrator")
    print("=" * 40)
    print("1. Setup (פעם ראשונה)")
    print("2. Run Orchestrator")
    print("3. Install Requirements")
    print("4. Exit")

    choice = input("\nבחר (1-4): ").strip()

    if choice == "1":
        print("\n🔧 מריץ Setup...")
        subprocess.run([sys.executable, "setup.py"])

    elif choice == "2":
        print("\n🚀 מריץ Orchestrator...")
        subprocess.run([sys.executable, "orchestrator.py"])

    elif choice == "3":
        print("\n📦 מתקין Requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    elif choice == "4":
        print("👋 יום טוב!")

    else:
        print("❌ בחירה לא תקינה")

if __name__ == "__main__":
    main()