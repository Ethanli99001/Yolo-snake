// Imports
import * as ort from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import * as ImageManipulator from "expo-image-manipulator";
import * as FileSystem from "expo-file-system/legacy";
import jpeg from "jpeg-js";

// Labels
const LABELS = [
  "Agkistrodon_Piscivorus",
  "Agkistrodon_contortrix",
  "Crotalus_Horridus",
  "Crotalus_adamanteus",
  "Micrurus_Fulvius",
  "Sistrurus_Miliarius",
];

// Image Size
const INPUT_SIZE = 640;

// ONNX Session
let sessionPromise: Promise<ort.InferenceSession> | null = null;
async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const asset = Asset.fromModule(
        require("../../assets/yolo26n.onnx")
      );

      await asset.downloadAsync();

      if (!asset.localUri) {
        throw new Error("Could not load the ONNX model asset");
      }

      return ort.InferenceSession.create(asset.localUri);
    })().catch((error) => {
      // Reattempt Session
      sessionPromise = null;
      throw error;
    });
  }

  return sessionPromise;
}

// Detection
export type Detection = {
  label: string;
  confidence: number;
  classId: number;
  box: [number, number, number, number];
};

// Run Model
export async function detectObjects(
  photoUri: string
): Promise<Detection[]> {
  const session = await getSession();

  // Preprocess
  // Resize image to 640 x 640
  const resized = await ImageManipulator.manipulateAsync(
    photoUri,
    [{ resize: { width: INPUT_SIZE, height: INPUT_SIZE } }],
    {
      compress: 1,
      format: ImageManipulator.SaveFormat.JPEG,
    }
  );

  // JPEG (Base64) to Bytes to RGBA
  const base64 = await FileSystem.readAsStringAsync(
    resized.uri,
    { encoding: FileSystem.EncodingType.Base64 }
  );

  const binary = globalThis.atob(base64);
  const jpegBytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    jpegBytes[i] = binary.charCodeAt(i);
  }

  const decoded = jpeg.decode(jpegBytes, {
    useTArray: true,
  });

  // Decoded JPEG (Width, Height, RGBA Data)
  const { width, height, data } = decoded;

  // Image Size Check
  if (width !== INPUT_SIZE || height !== INPUT_SIZE) {
    throw new Error(
      `Unexpected image size: ${width} x ${height}`
    );
  }

  // JPEG Interleaved RGBA to Sorted Normalized RGB for YOLO26
  const input = new Float32Array(
    1 * 3 * INPUT_SIZE * INPUT_SIZE
  );

  const plane = INPUT_SIZE * INPUT_SIZE;

  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const pixel = y * INPUT_SIZE + x;
      const rgba = pixel * 4;

      input[pixel] = data[rgba] / 255;
      input[plane + pixel] = data[rgba + 1] / 255;
      input[2 * plane + pixel] = data[rgba + 2] / 255;
    }
  }

  // ONNX Tensor
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const tensor = new ort.Tensor(
    "float32",
    input,
    [1, 3, INPUT_SIZE, INPUT_SIZE]
  );

  // Run YOLO26
  const outputs = await session.run({
    [inputName]: tensor,
  });

  // YOLO26 Output
  const output = outputs[outputName];

  // YOLO26 Output Check
  if (!output) {
    throw new Error(`Missing model output: ${outputName}`);
  }

  // YOLO26 nms=False Detection Check
  if (
    output.dims.length !== 3 ||
    output.dims[2] !== 6
  ) {
    throw new Error(
      `Unexpected YOLO output shape: ${output.dims.join(" x ")}. ` +
      "This helper expects an nms=False detection model."
    );
  }

  // Detection Data
  const values = output.data as Float32Array;
  const rows = output.dims[1];
  const detections: Detection[] = [];

  for (let i = 0; i < rows; i++) {
    const offset = i * 6;

    const x1 = values[offset];
    const y1 = values[offset + 1];
    const x2 = values[offset + 2];
    const y2 = values[offset + 3];
    const confidence = values[offset + 4];
    const classId = Math.round(values[offset + 5]);

    // Low Confidence Filter
    if (
      !Number.isFinite(confidence) ||
      confidence < 0.25 ||
      classId < 0 ||
      classId >= LABELS.length
    ) {
      continue;
    }

    // Add Detection
    detections.push({
      label: LABELS[classId],
      confidence,
      classId,
      box: [x1, y1, x2, y2],
    });
  }

  // Sort Detections by Confidence
  detections.sort((a, b) => b.confidence - a.confidence);

  // Return Detections
  return detections;
}
