const { Node, TM } = require("./index");

class Xornet {
    constructor(options = {}) {
        // 2->2->1 xornet
        this.w1 = new Node(options.w1 || [
            [Math.random() * 2 - 1, Math.random() * 2 - 1],
            [Math.random() * 2 - 1, Math.random() * 2 - 1]
        ]);
        this.b1 = new Node(options.b1 || [0, 0]);
        this.w2 = new Node(options.w2 || [
            [Math.random() * 2 - 1],
            [Math.random() * 2 - 1]
        ]);
        this.b2 = new Node(options.b2 || [0]);
        this.lr = options.lr || 0.5;
    }

    forward(input) {
        return new Node(input)
                    .matmul(this.w1)
                    .add(this.b1)
                    .sigmoid()
                    .matmul(this.w2)
                    .add(this.b2)
                    .sigmoid();
    }

    backprop(input, target) {
        const T = new Node(target);
        const Y = this.forward(input);
        const L = Y.sub(T).pow(2).mul(0.5);

        L.backward();

        this.w1.value = TM.sub(this.w1.value, TM.mul(this.w1.grad, this.lr));
        this.w2.value = TM.sub(this.w2.value, TM.mul(this.w2.grad, this.lr));
        this.b1.value = TM.sub(this.b1.value, TM.mul(this.b1.grad, this.lr));
        this.b2.value = TM.sub(this.b2.value, TM.mul(this.b2.grad, this.lr));
    }
}

const xornet = new Xornet();

for (let epoch = 0; epoch < 30000; epoch++) {
    xornet.backprop([1,0], [1]);
    xornet.backprop([0,1], [1]);
    xornet.backprop([0,0], [0]);
    xornet.backprop([1,1], [0]);
}

console.log(xornet.forward([1,1]).value); // 0-ish
console.log(xornet.forward([1,0]).value); // 1-ish
console.log(xornet.forward([0,1]).value); // 1-ish
console.log(xornet.forward([0,0]).value); // 0-ish
