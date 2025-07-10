"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TensorMath = void 0;
class TensorMath {
    static create(num, shape) {
        if (shape.length === 0) {
            return num;
        }
        const [dim, ...rest] = shape;
        const out = [];
        for (let i = 0; i < dim; i++) {
            out.push(TensorMath.create(num, rest));
        }
        return out;
    }
    static getShape(tA) {
        const shape = [];
        let subA = tA;
        while (Array.isArray(subA)) {
            shape.push(subA.length);
            subA = subA[0];
        }
        return shape;
    }
    static padShape(tA, tB) {
        let dimA = TensorMath.getShape(tA).length;
        let dimB = TensorMath.getShape(tB).length;
        while (dimA < dimB) {
            dimA++;
            tA = [tA];
        }
        while (dimA > dimB) {
            dimB++;
            tB = [tB];
        }
        return [tA, tB];
    }
    static elementWiseAB(tA, tB, op) {
        if (typeof tA === "number" && typeof tB === "number") {
            return op(tA, tB);
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.elementWiseAB(subA, subB, op));
        }
        return result;
    }
    static elementWiseSelf(tA, op) {
        if (typeof tA === "number") {
            return op(tA);
        }
        else {
            return tA.map(subA => TensorMath.elementWiseSelf(subA, op));
        }
    }
    static add(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA + tB);
    }
    static sub(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA - tB);
    }
    static mul(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA * tB);
    }
    static pow(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA ** tB);
    }
    static div(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA / tB);
    }
    static gt(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA > tB ? 1 : 0);
    }
    static lt(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA < tB ? 1 : 0);
    }
    static ge(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA >= tB ? 1 : 0);
    }
    static le(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA <= tB ? 1 : 0);
    }
    static eq(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA === tB ? 1 : 0);
    }
    static logicalAnd(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA === 1 && tB === 1 ? 1 : 0);
    }
    static logicalOr(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA === 1 || tB === 1 ? 1 : 0);
    }
    static logicalXor(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => (tA === 1 || tB === 1) && tA !== tB ? 1 : 0);
    }
    static logicalNot(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => tA === 1 ? 0 : 1);
    }
    static bitwiseAnd(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA & tB);
    }
    static bitwiseOr(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA | tB);
    }
    static bitwiseXor(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA ^ tB);
    }
    static bitwiseNot(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => ~tA);
    }
    static bitwiseLeftShift(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA << tB);
    }
    static bitwiseRightShift(tA, tB) {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA >> tB);
    }
    static neg(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => -tA);
    }
    static abs(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.abs(tA));
    }
    static sign(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sign(tA));
    }
    static sin(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sin(tA));
    }
    static cos(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.cos(tA));
    }
    static tan(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.tan(tA));
    }
    static asin(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.asin(tA));
    }
    static acos(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.acos(tA));
    }
    static atan(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.atan(tA));
    }
    static sinh(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sinh(tA));
    }
    static cosh(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.cosh(tA));
    }
    static asinh(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.asinh(tA));
    }
    static acosh(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.acosh(tA));
    }
    static atanh(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.atanh(tA));
    }
    static sqrt(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sqrt(tA));
    }
    static exp(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.exp(tA));
    }
    static log(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log(tA));
    }
    static log2(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log2(tA));
    }
    static log10(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log10(tA));
    }
    static log1p(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log(tA));
    }
    static relu(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.max(tA, 0));
    }
    static sigmoid(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => 1 / (1 + Math.exp(-tA)));
    }
    static tanh(tA) {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.tanh(tA));
    }
    static squeezeAxis(tA, axis) {
        if (typeof tA === "number")
            return tA;
        if (axis === 0) {
            return tA[0];
        }
        else {
            return tA.map(slice => TensorMath.squeezeAxis(slice, axis - 1));
        }
    }
    static squeeze(tA, dims) {
        if (typeof tA === "number")
            return tA;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            const shape = TensorMath.getShape(tA);
            dims = [];
            for (let index = 0; index < shape.length; index++) {
                if (shape[index] === 1) {
                    dims.push(index);
                }
            }
        }
        dims = [...dims].sort((a, b) => b - a);
        let out = tA;
        for (const axis of dims) {
            out = TensorMath.squeezeAxis(out, axis);
        }
        return out;
    }
    static sumAxis(tA, axis) {
        if (typeof tA === "number")
            return tA;
        if (axis === 0) {
            let result = tA[0];
            for (let i = 1; i < tA.length; i++) {
                result = TensorMath.add(result, tA[i]);
            }
            return [result];
        }
        else {
            return tA.map(slice => TensorMath.sumAxis(slice, axis - 1));
        }
    }
    static sum(tA, dims, keepDims = false) {
        if (typeof tA === "number")
            return tA;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: TensorMath.getShape(tA).length }, (_, index) => index);
        }
        dims = [...dims].sort((a, b) => b - a);
        let out = tA;
        for (const axis of dims) {
            out = TensorMath.sumAxis(out, axis);
        }
        return keepDims ? out : TensorMath.squeeze(out, dims);
    }
    static t(tA) {
        const shapeA = TensorMath.getShape(tA);
        if (shapeA.length !== 2)
            throw new Error("Input is not a matrix");
        const matA = tA;
        const matARows = matA.length;
        const matACols = matA[0].length;
        const matATranspose = Array.from({ length: matACols }, () => new Array(matARows).fill(0));
        for (let i = 0; i < matARows; i++) {
            for (let j = 0; j < matACols; j++) {
                matATranspose[j][i] = matA[i][j];
            }
        }
        return matATranspose;
    }
    static mm(tA, tB) {
        const shapeA = TensorMath.getShape(tA);
        const shapeB = TensorMath.getShape(tB);
        if (shapeA.length !== 2 || shapeB.length !== 2)
            throw new Error("Inputs are not matrices");
        const matA = tA;
        const matB = tB;
        const matARows = matA.length;
        const matACols = matA[0].length;
        const matBRows = matB.length;
        const matBCols = matB[0].length;
        if (matACols !== matBRows)
            throw new Error("Invalid matrices shape for multiplication");
        const matC = Array.from({ length: matARows }, () => new Array(matBCols).fill(0));
        for (let i = 0; i < matARows; i++) {
            for (let j = 0; j < matBCols; j++) {
                for (let k = 0; k < matACols; k++) {
                    matC[i][j] += matA[i][k] * matB[k][j];
                }
            }
        }
        return matC;
    }
    static dot(tA, tB) {
        const shapeA = TensorMath.getShape(tA);
        const shapeB = TensorMath.getShape(tB);
        if (shapeA.length !== 1 || shapeB.length !== 1 || shapeA[0] !== shapeB[0])
            throw new Error("Inputs are not 1D tensors");
        const vectLen = shapeA[0];
        const vectA = tA;
        const vectB = tB;
        let sum = 0;
        for (let index = 0; index < vectLen; index++) {
            sum += vectA[index] * vectB[index];
        }
        return sum;
    }
}
exports.TensorMath = TensorMath;
