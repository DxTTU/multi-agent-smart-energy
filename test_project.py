from agents import SmartEnergySystem

result = SmartEnergySystem().run()
assert "decision" in result
assert "rag_evidence" in result
assert "action" in result
print("Project 2 smoke test passed.")
