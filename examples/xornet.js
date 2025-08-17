const { Tensor, Optim, nn } = require("../index");

// Our model
class Xornet {
    constructor() {
        // 2->2->1 xornet
        this.l1 = new nn.Linear(2, 2);
        this.l2 = new nn.Linear(2, 1);
    }

    forward(input) {
        return this.l2.forward(this.l1.forward(new Tensor(input)).sigmoid()).sigmoid();
    }
}

// Our training loop
function train(model, epochs) {
    // Adam optimizer
    const optim = new Optim.Adam(nn.state.getParameters(model));

    // Batched training
    for (let epoch = 0; epoch < epochs; epoch++) {
        // Expected result
        const T = new Tensor([[1], [1], [0], [0]]);
        // Estimated result
        const Y = model.forward([[1, 0], [0, 1], [0, 0], [1, 1]]);
        // MSE loss
        const L = Y.sub(T).pow(2).mean();

        L.backward();

        optim.step();
    }
}

const xornet = new Xornet();

const start = performance.now()
train(xornet, 30000);
console.log(`Finished training in ${performance.now() - start} ms`);

console.log(xornet.forward([1,1]).val()); // 0-ish
console.log(xornet.forward([1,0]).val()); // 1-ish
console.log(xornet.forward([0,1]).val()); // 1-ish
console.log(xornet.forward([0,0]).val()); // 0-ish
