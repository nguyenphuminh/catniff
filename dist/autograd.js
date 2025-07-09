"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Node = exports.OP = void 0;
const tensor_1 = require("./tensor");
const { add, sub, mul, pow, div, gt, lt, ge, le, eq, neg, abs, sign, sin, cos, tan, asin, acos, atan, sinh, cosh, asinh, acosh, atanh, sqrt, exp, log, log2, log10, log1p, relu, sigmoid, tanh, t, mm } = tensor_1.TensorMath;
var OP;
(function (OP) {
    OP[OP["NONE"] = 0] = "NONE";
    OP[OP["ADD"] = 1] = "ADD";
    OP[OP["SUB"] = 2] = "SUB";
    OP[OP["MUL"] = 3] = "MUL";
    OP[OP["POW"] = 4] = "POW";
    OP[OP["DIV"] = 5] = "DIV";
    OP[OP["GE"] = 6] = "GE";
    OP[OP["LE"] = 7] = "LE";
    OP[OP["GT"] = 8] = "GT";
    OP[OP["LT"] = 9] = "LT";
    OP[OP["EQ"] = 10] = "EQ";
    OP[OP["NEG"] = 11] = "NEG";
    OP[OP["ABS"] = 12] = "ABS";
    OP[OP["SIGN"] = 13] = "SIGN";
    OP[OP["SIN"] = 14] = "SIN";
    OP[OP["COS"] = 15] = "COS";
    OP[OP["TAN"] = 16] = "TAN";
    OP[OP["ASIN"] = 17] = "ASIN";
    OP[OP["ACOS"] = 18] = "ACOS";
    OP[OP["ATAN"] = 19] = "ATAN";
    OP[OP["SINH"] = 20] = "SINH";
    OP[OP["COSH"] = 21] = "COSH";
    OP[OP["ASINH"] = 22] = "ASINH";
    OP[OP["ACOSH"] = 23] = "ACOSH";
    OP[OP["ATANH"] = 24] = "ATANH";
    OP[OP["SQRT"] = 25] = "SQRT";
    OP[OP["EXP"] = 26] = "EXP";
    OP[OP["LOG"] = 27] = "LOG";
    OP[OP["LOG2"] = 28] = "LOG2";
    OP[OP["LOG10"] = 29] = "LOG10";
    OP[OP["LOG1P"] = 30] = "LOG1P";
    OP[OP["RELU"] = 31] = "RELU";
    OP[OP["SIGMOID"] = 32] = "SIGMOID";
    OP[OP["TANH"] = 33] = "TANH";
    OP[OP["T"] = 34] = "T";
    OP[OP["MM"] = 35] = "MM";
})(OP || (exports.OP = OP = {}));
class Node {
    value;
    shape;
    grad;
    children;
    op;
    feedBackward;
    constructor(value, children = [], op = OP.NONE) {
        this.value = value;
        this.shape = tensor_1.TensorMath.getShape(value);
        this.grad = tensor_1.TensorMath.create(0, this.shape);
        this.children = children;
        this.op = op;
        this.feedBackward = () => { };
    }
    add(other) {
        other = Node.forceNode(other);
        const out = new Node(add(this.value, other.value), [this, other], OP.ADD);
        out.feedBackward = () => {
            // x + y d/dx = 1, note that we apply the chain rule continuously so out.grad is multiplied into our derivative
            Node.addGrad(this, out.grad);
            // x + y d/dy = 1
            Node.addGrad(other, out.grad);
        };
        return out;
    }
    sub(other) {
        other = Node.forceNode(other);
        const out = new Node(sub(this.value, other.value), [this, other], OP.SUB);
        out.feedBackward = () => {
            // x - y d/dx = 1
            Node.addGrad(this, out.grad);
            // x - y d/dy = -1
            Node.addGrad(other, neg(out.grad));
        };
        return out;
    }
    mul(other) {
        other = Node.forceNode(other);
        const out = new Node(mul(this.value, other.value), [this, other], OP.MUL);
        out.feedBackward = () => {
            // x * y d/dx = y
            Node.addGrad(this, mul(out.grad, other.value));
            // x * y d/dy = x
            Node.addGrad(other, mul(out.grad, this.value));
        };
        return out;
    }
    pow(other) {
        if (other instanceof Node) {
            const out = new Node(pow(this.value, other.value), [this, other], OP.POW);
            out.feedBackward = () => {
                // x^a d/dx = a*x^(a-1)
                Node.addGrad(this, mul(out.grad, mul(other.value, pow(this.value, sub(other.value, 1)))));
                // x^a d/da = x^a*lnx
                Node.addGrad(other, mul(out.grad, mul(pow(this.value, other.value), log(this.value))));
            };
            return out;
        }
        const out = new Node(pow(this.value, other), [this], OP.POW);
        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, mul(other, pow(this.value, sub(other, 1)))));
        };
        return out;
    }
    div(other) {
        other = Node.forceNode(other);
        const out = new Node(div(this.value, other.value), [this, other], OP.DIV);
        out.feedBackward = () => {
            // x/y d/dx = 1/y
            Node.addGrad(this, div(out.grad, other.value));
            // x/y d/dy = -x/y^2
            Node.addGrad(other, mul(out.grad, div(neg(this.value), pow(other.value, 2))));
        };
        return out;
    }
    ge(other) {
        other = Node.forceNode(other);
        const out = new Node(ge(this.value, other.value), [this, other], OP.GE);
        out.feedBackward = () => {
            // We consider the derivative of ge to be 0, which does not add to current grad, so this function is just empty
        };
        return out;
    }
    le(other) {
        other = Node.forceNode(other);
        const out = new Node(le(this.value, other.value), [this, other], OP.LE);
        out.feedBackward = () => {
            // We consider the derivative of le to be 0, which does not add to current grad, so this function is just empty
        };
        return out;
    }
    gt(other) {
        other = Node.forceNode(other);
        const out = new Node(gt(this.value, other.value), [this, other], OP.GT);
        out.feedBackward = () => {
            // We consider the derivative of gt to be 0, which does not add to current grad, so this function is just empty
        };
        return out;
    }
    lt(other) {
        other = Node.forceNode(other);
        const out = new Node(lt(this.value, other.value), [this, other], OP.LT);
        out.feedBackward = () => {
            // We consider the derivative of lt to be 0, which does not add to current grad, so this function is just empty
        };
        return out;
    }
    eq(other) {
        other = Node.forceNode(other);
        const out = new Node(eq(this.value, other.value), [this, other], OP.EQ);
        out.feedBackward = () => {
            // We consider the derivative of eq to be 0, which does not add to current grad, so this function is just empty
        };
        return out;
    }
    neg() {
        const out = new Node(neg(this.value), [this], OP.NEG);
        out.feedBackward = () => {
            // -x d/dx = -1
            Node.addGrad(this, neg(out.grad));
        };
        return out;
    }
    abs() {
        const out = new Node(abs(this.value), [this], OP.ABS);
        out.feedBackward = () => {
            // |x| d/dx = sign(x)
            Node.addGrad(this, mul(out.grad, sign(this.value)));
        };
        return out;
    }
    sign() {
        const out = new Node(sign(this.value), [this], OP.SIGN);
        out.feedBackward = () => {
            // We consider the derivative of sign to be 0, which does not add to current grad, so this function is just empty
        };
        return out;
    }
    sin() {
        const out = new Node(sin(this.value), [this], OP.SIN);
        out.feedBackward = () => {
            // sinx d/dx = cosx
            Node.addGrad(this, mul(out.grad, cos(this.value)));
        };
        return out;
    }
    cos() {
        const out = new Node(cos(this.value), [this], OP.COS);
        out.feedBackward = () => {
            // cosx d/dx = -sinx
            Node.addGrad(this, mul(out.grad, neg(sin(this.value))));
        };
        return out;
    }
    tan() {
        const tanResult = tan(this.value);
        const out = new Node(tanResult, [this], OP.TAN);
        out.feedBackward = () => {
            // tanx d/dx = 1+(tanx)^2
            Node.addGrad(this, mul(out.grad, add(1, pow(tanResult, 2))));
        };
        return out;
    }
    asin() {
        const out = new Node(asin(this.value), [this], OP.ASIN);
        out.feedBackward = () => {
            // asinx d/dx = 1/sqrt(1-x^2)
            Node.addGrad(this, div(out.grad, sqrt(sub(1, pow(this.value, 2)))));
        };
        return out;
    }
    acos() {
        const out = new Node(acos(this.value), [this], OP.ACOS);
        out.feedBackward = () => {
            // acosx d/dx = -1/sqrt(1-x^2)
            Node.addGrad(this, neg(div(out.grad, sqrt(sub(1, pow(this.value, 2))))));
        };
        return out;
    }
    atan() {
        const out = new Node(atan(this.value), [this], OP.ATAN);
        out.feedBackward = () => {
            // atanx d/dx = 1/(1+x^2)
            Node.addGrad(this, div(out.grad, add(1, pow(this.value, 2))));
        };
        return out;
    }
    sinh() {
        const out = new Node(sinh(this.value), [this], OP.SINH);
        out.feedBackward = () => {
            // sinhx d/dx = coshx
            Node.addGrad(this, mul(out.grad, cosh(this.value)));
        };
        return out;
    }
    cosh() {
        const out = new Node(cosh(this.value), [this], OP.COSH);
        out.feedBackward = () => {
            // coshx d/dx = sinhx
            Node.addGrad(this, mul(out.grad, sinh(this.value)));
        };
        return out;
    }
    asinh() {
        const out = new Node(asinh(this.value), [this], OP.ASINH);
        out.feedBackward = () => {
            // asinhx d/dx = 1/sqrt(1+x^2)
            Node.addGrad(this, div(out.grad, sqrt(add(1, pow(this.value, 2)))));
        };
        return out;
    }
    acosh() {
        const out = new Node(acosh(this.value), [this], OP.ACOSH);
        out.feedBackward = () => {
            // acosx d/dx = 1/(sqrt(x-1)*sqrt(x+1))
            Node.addGrad(this, div(out.grad, mul(sqrt(sub(this.value, 1)), sqrt(add(this.value, 1)))));
        };
        return out;
    }
    atanh() {
        const out = new Node(atanh(this.value), [this], OP.ATANH);
        out.feedBackward = () => {
            // atanx d/dx = 1/(1-x^2)
            Node.addGrad(this, div(out.grad, sub(1, pow(this.value, 2))));
        };
        return out;
    }
    sqrt() {
        const sqrtResult = sqrt(this.value);
        const out = new Node(sqrtResult, [this], OP.SQRT);
        out.feedBackward = () => {
            // sqrt(x) d/dx = 1/(2*sqrt(x))
            Node.addGrad(this, div(out.grad, mul(2, sqrtResult)));
        };
        return out;
    }
    exp() {
        const expResult = exp(this.value);
        const out = new Node(expResult, [this], OP.EXP);
        out.feedBackward = () => {
            // e^x d/dx = e^x
            Node.addGrad(this, mul(out.grad, expResult));
        };
        return out;
    }
    log() {
        const out = new Node(log(this.value), [this], OP.LOG);
        out.feedBackward = () => {
            // lnx d/dx = 1/x
            Node.addGrad(this, div(out.grad, this.value));
        };
        return out;
    }
    log2() {
        const out = new Node(log2(this.value), [this], OP.LOG2);
        out.feedBackward = () => {
            // log2(x) d/dx = 1/(xln2)
            Node.addGrad(this, div(out.grad, mul(this.value, Math.log(2))));
        };
        return out;
    }
    log10() {
        const out = new Node(log10(this.value), [this], OP.LOG10);
        out.feedBackward = () => {
            // log2(x) d/dx = 1/(xln10)
            Node.addGrad(this, div(out.grad, mul(this.value, Math.log(10))));
        };
        return out;
    }
    log1p() {
        const out = new Node(log1p(this.value), [this], OP.LOG1P);
        out.feedBackward = () => {
            // ln(1+x) d/dx = 1/(1+x)
            Node.addGrad(this, div(out.grad, add(this.value, 1)));
        };
        return out;
    }
    relu() {
        const out = new Node(relu(this.value), [this], OP.RELU);
        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, ge(this.value, 0)));
        };
        return out;
    }
    sigmoid() {
        const sigmoidResult = sigmoid(this.value);
        const out = new Node(sigmoidResult, [this], OP.SIGMOID);
        out.feedBackward = () => {
            Node.addGrad(this, mul(mul(out.grad, sigmoidResult), sub(1, sigmoidResult)));
        };
        return out;
    }
    tanh() {
        const tanhResult = tanh(this.value);
        const out = new Node(tanhResult, [this], OP.TANH);
        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, sub(1, mul(tanhResult, tanhResult))));
        };
        return out;
    }
    t() {
        const out = new Node(t(this.value), [this], OP.T);
        out.feedBackward = () => {
            Node.addGrad(this, t(out.grad));
        };
        return out;
    }
    mm(other) {
        other = Node.forceNode(other);
        const out = new Node(mm(this.value, other.value), [this, other], OP.MM);
        out.feedBackward = () => {
            Node.addGrad(this, mm(out.grad, t(other.value)));
            Node.addGrad(other, mm(t(this.value), out.grad));
        };
        return out;
    }
    backward() {
        // Build topological order
        const topo = [];
        const visited = new Set();
        function build(node) {
            if (!visited.has(node)) {
                visited.add(node);
                node.grad = tensor_1.TensorMath.create(0, node.shape);
                for (let child of node.children)
                    build(child);
                topo.push(node);
            }
        }
        build(this);
        // Feed backward to calculate gradient
        this.grad = tensor_1.TensorMath.create(1, this.shape); // Derivative of itself with respect to itself
        for (let index = topo.length - 1; index > -1; index--) {
            topo[index].feedBackward();
        }
    }
    static forceNode(value) {
        if (value instanceof Node)
            return value;
        return new Node(value);
    }
    static addGrad(node, accumGrad) {
        const axesToSqueeze = [];
        const axesToReduce = [];
        const shape = node.shape;
        const gradShape = tensor_1.TensorMath.getShape(accumGrad);
        const paddedDims = gradShape.length - shape.length;
        for (let i = 0; i < paddedDims; i++) {
            axesToReduce.push(i);
            axesToSqueeze.push(i);
        }
        for (let i = 0; i < shape.length; i++) {
            if (shape[i] === 1 && gradShape[i + paddedDims] > 1) {
                axesToReduce.push(i + paddedDims);
            }
        }
        const reducedGrad = tensor_1.TensorMath.sum(accumGrad, axesToReduce, true);
        const squeezedGrad = tensor_1.TensorMath.squeeze(reducedGrad, axesToSqueeze);
        node.grad = add(squeezedGrad, node.grad);
    }
}
exports.Node = Node;
