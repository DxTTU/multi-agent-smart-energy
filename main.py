from agents import SmartEnergySystem

if __name__ == "__main__":
    result = SmartEnergySystem().run()
    print("=== MULTI-AGENT SMART ENERGY SYSTEM ===")
    print(f"Current load: {result['current_kwh']} kWh")
    print(f"Forecast load: {result['forecast_kwh']} kWh")
    print(f"Policy risk: {result['policy_risk']}")
    print(f"Decision: {result['decision'].action}")
    print(f"Reason: {result['decision'].reason}")
    print("\nRAG evidence:")
    for item in result["rag_evidence"]:
        print("-", item.replace("\n", " "))
    print("\nController:")
    print(result["action"])
