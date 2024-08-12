import { NeuralNetwork } from "./nn";
import { convertToCSV, generateSpiralDataset, revertData, type SpiralDataPoint } from "./data";

// Generate the spiral dataset
const { trainingData, testingData } = generateSpiralDataset(10, 9);

// Convert the dataset to the format required by the neural network
function convertData(data: SpiralDataPoint[]): {
  inputs: number[][][];
  targets: number[][][];
} {
  const inputs: number[][][] = [];
  const targets: number[][][] = [];

  data.forEach((point) => {
    inputs.push([[point.x], [point.y]]);
    const target = Array(9)
      .fill(0)
      .map(() => [0]); // One-hot encoding for 3 classes
    target[point.class][0] = 1;
    targets.push(target);
  });

  return { inputs, targets };
}

const trainingSet = convertData(trainingData);
const testingSet = convertData(testingData);

// Initialize the neural network
const nn = new NeuralNetwork([2, 10, 9]);

// Train the neural network
nn.train(trainingSet.inputs, trainingSet.targets, 10000, 0.01);

// Test the neural network on the testing data
let correctPredictions = 0;
// Initialize predictions object
const predictions: ReturnType<typeof convertData> = {
  inputs: [],
  targets: [],
};

testingSet.inputs.forEach((input, index) => {
  const prediction = nn.predict(input);
  const actualClass = testingSet.targets[index].findIndex((t) => t[0] === 1);
  const predictedClass = prediction.findIndex(
    (p) => p[0] === Math.max(...prediction.map((v) => v[0]))
  );

  console.log(
    `Input: [${input[0][0]}, ${input[1][0]}], Predicted Class: ${predictedClass}, Actual Class: ${actualClass}`
  );

  if (predictedClass === actualClass) {
    correctPredictions++;
  }

  // Store the input and the predicted target in the predictions object
  predictions.inputs.push(input);
  const predictedTarget = Array(9)
    .fill(0)
    .map(() => [0]); // One-hot encoding for 3 classes
  predictedTarget[predictedClass][0] = 1;
  predictions.targets.push(predictedTarget);
});

// Calculate accuracy
const accuracy = (correctPredictions / testingSet.inputs.length) * 100;
console.log(`Accuracy: ${accuracy.toFixed(2)}%`);

const predictionData = revertData(predictions.inputs, predictions.targets);
const testingCSV = convertToCSV(testingData);
const resultCSV = convertToCSV(predictionData);
Bun.write("./testing.csv", testingCSV).then(console.log)
Bun.write("./result.csv", resultCSV).then(console.log)