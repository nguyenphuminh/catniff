const { Tensor, Optim } = require("../index");

// Define some parameter
const w = new Tensor([1.0], { requiresGrad: true });
// Define a fake loss function: L = (w - 3)^2
const loss = w.sub(3).pow(2);
// Calculate gradient
loss.backward();
// Use Adam optimizer
const optim = new Optim.Adam([w]);
// Optimization step
optim.step();

console.log(`Updated weight: ${w}`);  // Should move toward 3.0
