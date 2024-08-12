import { convertToCSV, generateSpiralDataset } from "./data";

const data = generateSpiralDataset(100, 3);

const csv = convertToCSV(data);

Bun.write("./data.csv", csv).then(() => console.log("done"))