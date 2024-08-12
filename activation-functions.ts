import { maths } from "utiliti-js";
import { Matrix } from "./matrix";

export class Activation {
  static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  static sigmoidDerivative(x: number): number {
    const sig = Activation.sigmoid(x);
    return sig * (1 - sig);
  }

  static relu(x: number): number {
    return Math.max(0, x);
  }

  static reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }
}

export function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

export function relu(x: number) {
  return Math.max(0, x);
}

export function softmax(inputs: number[][]) {
  const outputs: number[][] = [];

  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];
    outputs[i] = [];
    for (let j = 0; j < input.length; j++) {
      const max = Math.max(0, ...input);
      outputs[i][j] = Math.exp(input[j] - max);
    }

    const output = outputs[i];
    const sum_out = output.reduce((a, b) => a + b, 0);
    outputs[i] = output.map((v) => v / sum_out);
  }

  return outputs;
}

export function categoricalCrossEntropy(
  predictions: number[][],
  targets: number[] | number[][]
): number[] {
  const predictionMatrix = Matrix.from(predictions);
  const clippedPredictions = predictionMatrix.map((v) =>
    maths.clamp(v, 1e-7, 1 - 1e-7)
  );

  if (Array.isArray(targets) && targets.every((n) => typeof n === "number")) {
    // Case: targets are scalar indices
    return (targets as number[]).map((targetIndex, sampleIndex) => {
      const samplePredictions = clippedPredictions.data[sampleIndex];
      return -maths.naturalLogarithm(samplePredictions[targetIndex]);
    });
  } else {
    // Case: targets are one-hot encoded vectors
    const targetMatrix = Matrix.from(targets as number[][]);
    return clippedPredictions.data.map((samplePredictions, sampleIndex) => {
      const sampleTargets = targetMatrix.data[sampleIndex];
      const summedLogLikelihoods = samplePredictions.reduce(
        (sum, pred, classIndex) => {
          return sum + sampleTargets[classIndex] * maths.naturalLogarithm(pred);
        },
        0
      );
      return -summedLogLikelihoods;
    });
  }
}

export class Loss {
  static mse(predicted: number[][], actual: number[][]): number {
    const errors = Matrix.subtract(predicted, actual);
    const squaredErrors = Matrix.applyFunction(errors, (x) => x * x);
    const sum = squaredErrors.flat().reduce((acc, val) => acc + val, 0);
    return sum / squaredErrors.length;
  }

  static mseDerivative(predicted: number[][], actual: number[][]): number[][] {
    return Matrix.scalarMultiply(
      Matrix.subtract(predicted, actual),
      2 / predicted.length
    );
  }
}
