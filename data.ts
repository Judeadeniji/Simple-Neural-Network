export interface SpiralDataPoint {
  x: number;
  y: number;
  class: number;
}

export function random(min?: number, max?: number): number {
  if (min === undefined && max === undefined) {
    return Math.random();
  }
  if (max === undefined) {
    max = min as number;
    min = 0;
  }
  if (min! > max) {
    [min, max] = [max, min];
  }
  const isInteger = min! % 1 === 0 && max! % 1 === 0;
  let randomValue = Math.random() * (max! - min! + 1) + min!;
  if (isInteger) {
    randomValue = Math.floor(randomValue);
  }
  return randomValue;
}

export function generateSpiralDataset(
  numPointsPerClass: number,
  numClasses: number
): { trainingData: SpiralDataPoint[]; testingData: SpiralDataPoint[] } {
  const data: SpiralDataPoint[] = [];
  const noise = 0.1; // Add some noise for better separation
  const bound = 5;

  for (let c = 0; c < numClasses; c++) {
    const classOffset = (c * 2 * Math.PI) / numClasses; // Offset angle for each class

    for (let i = 0; i < numPointsPerClass; i++) {
      const t = (i * 4 * Math.PI) / numPointsPerClass + classOffset; // Angle increases linearly with i
      const r = (i / numPointsPerClass) * bound; // Radius increases with i
      const x = r * Math.cos(t) + noise * random(-1, 1); // Add noise
      const y = r * Math.sin(t) + noise * random(-1, 1); // Add noise
      data.push({ x, y, class: c });
    }
  }

  // Shuffle the data
  const shuffledData = data.sort(() => Math.random() - 0.5);

  // Split data into training (80%) and testing (20%)
  const splitIndex = Math.floor(shuffledData.length * 0.8);
  const trainingData = shuffledData.slice(0, splitIndex);
  const testingData = shuffledData.slice(splitIndex);

  return { trainingData, testingData };
}

// Function to convert dataset to CSV format
export function convertToCSV(data: SpiralDataPoint[]): string {
  const headers = ["x", "y", "class"];
  const rows = data.map((d) => `${d.x},${d.y},${d.class}`);
  return [headers.join(","), ...rows].join("\n");
}

// Function to convert dataset to JSON format
export function convertToJSON(data: SpiralDataPoint[]): string {
  return JSON.stringify(data, null, 2);
}

export function revertData(
  inputs: number[][][], 
  targets: number[][][]
): SpiralDataPoint[] {
  const data: SpiralDataPoint[] = [];

  inputs.forEach((input, index) => {
      const x = input[0][0];
      const y = input[1][0];

      const target = targets[index];
      const classIndex = target.findIndex(t => t[0] === 1); // Find the class based on one-hot encoding

      data.push({ x, y, class: classIndex });
  });

  return data;
}
