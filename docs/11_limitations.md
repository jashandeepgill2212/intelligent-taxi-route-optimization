# Chapter 11: System Limitations

## 11.1 Technical Limitations
1. **Fixed Outgoing Action Degree**: The environment uses action masking over a fixed max outgoing degree ($K=8$). Nodes with $>8$ neighbors truncate excess outgoing edges.
2. **Simplified Traffic Simulation**: Edge traffic multipliers are simulated (LOW=1.0x, MEDIUM=1.3x, HIGH=1.8x) rather than connected to live real-time IoT traffic cameras.
3. **Graph Scale Constraints**: Deep Q-Networks require retaking action decisions at each intersection, scaling linearly with step length.
