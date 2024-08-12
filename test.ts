import { NeuralNetwork } from "./nn";

interface TextDataPoint {
  text: string;
  class: number; // 0 for Negative, 1 for Positive
}

function generateSentimentDataset(): TextDataPoint[] {
  return [
    { text: "I love this!", class: 1 },
    { text: "This is terrible.", class: 0 },
    { text: "I'm so happy.", class: 1 },
    { text: "I hate it.", class: 0 },
    { text: "This is the best day ever.", class: 1 },
    { text: "This is the worst day ever.", class: 0 },
    { text: "I'm so excited!", class: 1 },
    { text: "I'm really disappointed.", class: 0 },
  ];
}

// Function to convert text to a simple bag-of-words vector
function textToVector(text: string, vocabulary: string[]): number[][] {
  const words = text.toLowerCase().split(/\W+/);
  return vocabulary.map((word) => (words.includes(word) ? [1] : [0]));
}

// Function to convert the inputs and predictions to CSV for 3D plotting
function convertToCSVFor3DPlot(inputs: number[][][], predictions: number[][][]): string {
  const headers = ["x", "y", "z"]; // 'z' is the prediction or class
  const rows = inputs.map((input, index) => {
    const x = input[0][0]; // First word vector
    const y = input[1][0]; // Second word vector
    const z = predictions[index][0]; // Prediction probability for class 0 (or class 1 depending on the use case)
    return `${x},${y},${z[0]}`;
  });
  return [headers.join(","), ...rows].join("\n");
}

// Example usage:
const vocabulary = [
  "love",
  "terrible",
  "happy",
  "hate",
  "best",
  "worst",
  "excited",
  "disappointed",
];

const sentimentData = generateSentimentDataset();
const inputs = sentimentData.map((data) => textToVector(data.text, vocabulary));
const targets = sentimentData.map((data) => [
  data.class === 0 ? [1] : [0],
  data.class === 1 ? [1] : [0],
]);

const nn = new NeuralNetwork([vocabulary.length, 4, 2]);

// Train the network
nn.train(inputs, targets, 20000, 0.1);

// Predict and convert to CSV
const predictions = inputs.map((input) => nn.predict(input));

const csvData = convertToCSVFor3DPlot(inputs, predictions);

Bun.write("./testing.csv", csvData).then(console.log)

// Test the neural network with a new sentence
const testSentence = "I am so disappointed with this";
const testVector = textToVector(testSentence, vocabulary);
const testOutput = nn.predict(testVector);
console.log("Test Vector:", testVector);
console.log("Predicted Output for test sentence:", testOutput);

const resultCSV = convertToCSVFor3DPlot([testOutput], [[[1]], [[0]]])
Bun.write("./result.csv", resultCSV).then(console.log)