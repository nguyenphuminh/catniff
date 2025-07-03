const { Node } = require("../index");

const X = new Node([
    [ 0.5, -1.0 ],
    [ 2.0,  0.0 ]
]);

const Y = new Node([
    [ 1.0, -2.0 ],
    [ 0.5,  1.5 ]
]);

const D = X.sub(Y);
const E = D.exp();
const F = E.add(1);
const G = F.log();

G.backward();

console.log(X.grad, Y.grad);
