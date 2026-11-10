
// Imports
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import {
  detectObjects,
  type Detection,
} from "../../utils/yolo";


// Default Function
export default function CameraScreen() {

  // Variables
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<"front" | "back">("back");
  const camera = useRef<CameraView>(null);

  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Photo Function
  async function takePhoto() {
    if (!camera.current || loading) return;

    setLoading(true);
    setError(null);
    setDetections([]);

    try {
      const photo = await camera.current.takePictureAsync({
        quality: 0.9,
      });

      if (!photo?.uri) {
        throw new Error("Failed to capture photo.");
      }

      setPhotoUri(photo.uri);

      const results = await detectObjects(photo.uri);
      setDetections(results);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Inference failed."
      );
    } finally {
      setLoading(false);
    }
  }

  // Retake Photo Function
  function retakePhoto() {
    setPhotoUri(null);
    setDetections([]);
    setError(null);
  }

  // Permission Check (Camera)
  if (!permission) {
    return <View style={styles.container} />;
  }

  // Permission Check (Camera)
  if (!permission.granted) {
    return (
      <View style={styles.permission}>
        <Text>Camera permission is required.</Text>
        <TouchableOpacity onPress={requestPermission}>
          <Text style={styles.buttonText}>Allow Camera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // (Camera View || Result View) & (Controls)
  return (
    <View style={styles.container}>
      {photoUri ? (
        <Image
          source={{ uri: photoUri }}
          style={StyleSheet.absoluteFill}
          resizeMode="cover"
        />
      ) : (
        <CameraView
          ref={camera}
          style={StyleSheet.absoluteFill}
          facing={facing}
        />
      )}
      {/* Controls */}
      <View style={styles.controls}>
        {!photoUri ? (
          <>
           {/* Flip Camera Button */}
            <TouchableOpacity
              style={styles.sideButton}
              disabled={loading}
              onPress={() =>
                setFacing((current) =>
                  current === "back" ? "front" : "back"
                )
              }
            >
              <Image
                source={require('@/assets/images/rotate.png')}
                style={styles.icon}
                resizeMode="contain"
              />
            </TouchableOpacity>

            {/* Shutter Button */}
            <TouchableOpacity
              style={styles.shutterOuter}
              onPress={takePhoto}
              disabled={loading}
            >
              <View style={styles.shutterInner} />
            </TouchableOpacity>

            {/* Ghost Button To Center Shutter Button */}
            <View style={styles.sideButton} />
          </>
        ) : (
          // Return Home Button (IN RESULT PAGE)
          <TouchableOpacity
            style={styles.retakeButton}
            onPress={retakePhoto}
            disabled={loading}
          >
            <Text style={styles.buttonText}>  Return  </Text>
          </TouchableOpacity>
        )}
      </View>
      
      {/* Loading Result || Result */}
      {(loading || detections.length > 0 || error) && (
        <View style={styles.resultPanel}>
          {loading ? (
            <>
              <ActivityIndicator size="large" color="#fff" />
              <Text style={styles.resultText}>
                Running YOLO26
              </Text>
            </>
          ) : error ? (
            <Text style={styles.errorText}>{error}</Text>
          ) : detections.length === 0 ? (
            <Text style={styles.resultText}>
              No objects detected above the confidence threshold.
            </Text>
          ) : (
            <>
              <Text style={styles.heading}>Detection results</Text>

              {detections.slice(0, 5).map((item, index) => (
                <Text
                  key={`${item.classId}-${index}`}
                  style={styles.resultText}
                >
                  {item.label}: {(item.confidence * 100).toFixed(1)}%
                </Text>
              ))}
            </>
          )}
        </View>
      )}
    </View>
  );
}

// Style Sheet
// --------------------------------------------------------------------------------------
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "black",
  },
  permission: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 20,
  },
  controls: {
    position: "absolute",
    bottom: 24,
    width: "100%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-around",
  },
  shutterOuter: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 5,
    borderColor: "white",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 32,
  },
  shutterInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "white",
  },
  sideButton: {
    width: 32,
    height: 32,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 32,
    marginLeft: 8,
  },
  icon: {
    width: 32,
    height: 32,
    transform: [{ rotate: '45deg' }],
    tintColor: "#ffffffc1"
  },
  retakeButton: {
    backgroundColor: "#1f1f1f",
    borderRadius: 8,
    paddingHorizontal: 24,
    paddingVertical: 14,
    marginBottom: 32,
  },
  resultPanel: {
    position: "absolute",
    top: 55,
    left: 16,
    right: 16,
    backgroundColor: "rgba(0, 0, 0, 0.75)",
    borderRadius: 8,
    padding: 16,
    gap: 8,
  },
  heading: {
    color: "white",
    fontSize: 18,
    fontWeight: "bold",
  },
  resultText: {
    color: "white",
    fontSize: 16,
  },
  errorText: {
    color: "#ff8a8a",
    fontSize: 14,
  },
  buttonText: {
    color: "#fbfcfc",
    fontSize: 18,
  },
});
