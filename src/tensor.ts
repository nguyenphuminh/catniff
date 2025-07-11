export type Tensor = number | Tensor[];

export class TensorMath {
    static create(num: number, shape: number[]): Tensor {
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

    static getShape(tA: Tensor): number[] {
        const shape: number[] = [];

        let subA = tA;

        while (Array.isArray(subA)) {
            shape.push(subA.length);
            subA = subA[0];
        }

        return shape;
    }

    static padShape(tA: Tensor, tB: Tensor): [Tensor[], Tensor[]] {
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

        return [tA as Tensor[], tB as Tensor[]];
    }

    static elementWiseAB(tA: Tensor, tB: Tensor, op: (tA: number, tB: number) => number): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return op(tA, tB);
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);

        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }

        const result: Tensor[] = [];

        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.elementWiseAB(subA, subB, op));
        }

        return result;
    }

    static elementWiseSelf(tA: Tensor, op: (tA: number) => number): Tensor {
        if (typeof tA === "number") {
            return op(tA);
        } else {
            return tA.map(subA => TensorMath.elementWiseSelf(subA, op));
        }
    }

    static add(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA + tB);
    }

    static sub(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA - tB);
    }

    static mul(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA * tB);
    }

    static pow(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA ** tB);
    }

    static div(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA / tB);
    }

    static gt(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA > tB ? 1 : 0);
    }

    static lt(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA < tB ? 1 : 0);
    }

    static ge(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA >= tB ? 1 : 0);
    }

    static le(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA <= tB ? 1 : 0);
    }

    static eq(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA === tB ? 1 : 0);
    }

    static logicalAnd(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA === 1 && tB === 1 ? 1 : 0);
    }

    static logicalOr(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA === 1 || tB === 1 ? 1 : 0);
    }

    static logicalXor(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => (tA === 1 || tB === 1) && tA !== tB ? 1 : 0);
    }

    static logicalNot(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => tA === 1 ? 0 : 1);
    }

    static bitwiseAnd(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA & tB);
    }

    static bitwiseOr(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA | tB);
    }

    static bitwiseXor(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA ^ tB);
    }

    static bitwiseNot(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => ~tA);
    }

    static bitwiseLeftShift(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA << tB);
    }

    static bitwiseRightShift(tA: Tensor, tB: Tensor): Tensor {
        return TensorMath.elementWiseAB(tA, tB, (tA, tB) => tA >> tB);
    }

    static neg(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => -tA);
    }

    static abs(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.abs(tA));
    }

    static sign(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sign(tA));
    }

    static sin(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sin(tA));
    }

    static cos(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.cos(tA));
    }

    static tan(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.tan(tA));
    }

    static asin(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.asin(tA));
    }

    static acos(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.acos(tA));
    }

    static atan(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.atan(tA));
    }

    static sinh(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sinh(tA));
    }

    static cosh(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.cosh(tA));
    }

    static asinh(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.asinh(tA));
    }

    static acosh(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.acosh(tA));
    }

    static atanh(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.atanh(tA));
    }

    static sqrt(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.sqrt(tA));
    }

    static exp(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.exp(tA));
    }

    static log(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log(tA));
    }

    static log2(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log2(tA));
    }

    static log10(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log10(tA));
    }

    static log1p(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.log(tA));
    }

    static relu(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.max(tA, 0));
    }

    static sigmoid(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => 1 / (1 + Math.exp(-tA)));
    }

    static tanh(tA: Tensor): Tensor {
        return TensorMath.elementWiseSelf(tA, (tA) => Math.tanh(tA));
    }

    static squeezeAxis(tA: Tensor, axis: number): Tensor {
        if (typeof tA === "number") return tA;

        if (axis === 0) {
            return tA[0];
        } else {
            return tA.map(slice => TensorMath.squeezeAxis(slice, axis - 1));
        }
    }

    static squeeze(tA: Tensor, dims?: number[] | number): Tensor {
        if (typeof tA === "number") return tA;
        if (typeof dims === "number") { dims = [dims]; }
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

        let out: Tensor = tA;

        for (const axis of dims) {
            out = TensorMath.squeezeAxis(out, axis);
        }

        return out;
    }

    static sumAxis(tA: Tensor, axis: number): Tensor {
        if (typeof tA === "number") return tA;

        if (axis === 0) {
            let result = tA[0];

            for (let i = 1; i < tA.length; i++) {
                result = TensorMath.add(result, tA[i]);
            }

            return [result];
        } else {
            return tA.map(slice => TensorMath.sumAxis(slice as Tensor[], axis - 1));
        }
    }

    static sum(tA: Tensor, dims?: number[] | number, keepDims: boolean = false): Tensor {
        if (typeof tA === "number") return tA;
        if (typeof dims === "number") { dims = [dims]; }
        if (typeof dims === "undefined") { dims = Array.from({ length: TensorMath.getShape(tA).length }, (_, index) => index); }

        dims = [...dims].sort((a, b) => b - a);

        let out: Tensor = tA;

        for (const axis of dims) {
            out = TensorMath.sumAxis(out, axis);
        }

        return keepDims ? out : TensorMath.squeeze(out, dims);
    }

    static t(tA: Tensor): Tensor {
        const shapeA = TensorMath.getShape(tA);

        if (shapeA.length !== 2) throw new Error("Input is not a matrix");

        const matA = tA as number[][];
        const matARows = matA.length;
        const matACols = (matA[0] as Tensor[]).length;

        const matATranspose = Array.from({ length: matACols }, () => new Array(matARows).fill(0));

        for (let i = 0; i < matARows; i++) {
            for (let j = 0; j < matACols; j++) {
                matATranspose[j][i] = matA[i][j];
            }
        }

        return matATranspose;
    }

    static dot(tA: Tensor, tB: Tensor): Tensor {
        const shapeA = TensorMath.getShape(tA);
        const shapeB = TensorMath.getShape(tB);

        if (shapeA.length !== 1 || shapeB.length !== 1 || shapeA[0] !== shapeB[0]) throw new Error("Inputs are not 1D tensors");

        const vectLen = shapeA[0];
        const vectA = tA as number[];
        const vectB = tB as number[];
        let sum = 0;

        for (let index = 0; index < vectLen; index++) {
            sum += vectA[index] * vectB[index];
        }

        return sum;
    }

    static mm(tA: Tensor, tB: Tensor): Tensor {
        const shapeA = TensorMath.getShape(tA);
        const shapeB = TensorMath.getShape(tB);

        if (shapeA.length !== 2 || shapeB.length !== 2) throw new Error("Inputs are not matrices");

        const matA = tA as number[][];
        const matB = tB as number[][];
        const matARows = matA.length;
        const matACols = matA[0].length;
        const matBRows = matB.length;
        const matBCols = matB[0].length;

        if (matACols !== matBRows) throw new Error("Invalid matrices shape for multiplication");

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

    static mv(tA: Tensor, tB: Tensor): Tensor {
        const shapeA = TensorMath.getShape(tA);
        const shapeB = TensorMath.getShape(tB);

        if (shapeA.length !== 2 || shapeB.length !== 1) throw new Error("Input is not a 2D and 1D tensor pair");

        const matA = tA as number[][];
        const matB = (tB as number[]).map(el => [el]); // Turn the 1D tensor into a nx1 matrix (vector)

        return (TensorMath.mm(matA, matB) as number[][]).map(el => el[0]);

    }

    static matmul(tA: Tensor, tB: Tensor): Tensor {
        const shapeA = TensorMath.getShape(tA);
        const shapeB = TensorMath.getShape(tB);

        if (shapeA.length === 1 && shapeB.length === 1) {
            return TensorMath.dot(tA, tB);
        } else if (shapeA.length === 1 && shapeB.length === 2) {
            return (TensorMath.mm([tA], tB) as number[][])[0];
        } else if (shapeA.length === 2 && shapeB.length === 1) {
            return TensorMath.mv(tA, tB);
        } else if (shapeA.length === 2 && shapeB.length === 2) {
            return TensorMath.mm(tA, tB);
        }

        // Batched matmul will come when general nD transpose is done

        throw new Error(`Shapes [] and [] are not supported`);
    }
}

export const TM = TensorMath;
export const TMath = TensorMath;
