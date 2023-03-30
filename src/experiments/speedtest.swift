import CoreML

@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
class Input : MLFeatureProvider {

    /// Indices of input sequence tokens in the vocabulary as 1 by 128 matrix of 32-bit integers
    var input_ids: MLMultiArray

    var featureNames: Set<String> {
        get {
            return ["input_ids"]
        }
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "input_ids") {
            return MLFeatureValue(multiArray: input_ids)
        }
        return nil
    }

    init(input_ids: MLMultiArray) {
        self.input_ids = input_ids
    }

    convenience init(input_ids: MLShapedArray<Int32>) {
        self.init(input_ids: MLMultiArray(input_ids))
    }
}


/// Model Prediction Output Type
@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
class Output : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// Classification scores for each vocabulary token (after softmax) as 1 × 128 × 50257 3-dimensional array of floats
    var logits: MLMultiArray {
        return self.provider.featureValue(for: "logits")!.multiArrayValue!
    }

    /// Classification scores for each vocabulary token (after softmax) as 1 × 128 × 50257 3-dimensional array of floats
    var token_scoresShapedArray: MLShapedArray<Float> {
        return MLShapedArray<Float>(self.logits)
    }

    var featureNames: Set<String> {
        return self.provider.featureNames
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(logits: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["logits" : MLFeatureValue(multiArray: logits)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}

if CommandLine.argc == 1 {
    print("usage: \(CommandLine.arguments[0]) [model path]")
    print("compiles a model for faster load times")
    print("model path may be an .mlpackage, .neuralnetweork, or anything else CoreML understands")
    exit(1)
}
let modelPath = CommandLine.arguments[1]
var unitName = "All"
if (CommandLine.argc > 2) {
    unitName = CommandLine.arguments[2]
}
let modelURL = URL(filePath: modelPath)

let nameToUnit: Dictionary<String, MLComputeUnits> = [
    "All": .all,
    "CPUOnly": .cpuOnly,
    "CPUAndGPU": .cpuAndGPU,
    "CPUAndANE": .cpuAndNeuralEngine
]
let unit = nameToUnit[unitName]!

print("Loading model from \(modelURL.path) for compute unit \(unitName)")
let config = MLModelConfiguration()
config.computeUnits = unit
let model = try! MLModel(contentsOf: modelURL, configuration: config)
print("Loaded. Predicting...")

let loopCount: Int = 20
var total: Double = 0

(0..<loopCount).forEach { _ in
    let start = CFAbsoluteTimeGetCurrent()
    let input = Input(input_ids: MLShapedArray(scalars: Array(repeating: Int32(350), count: 512), shape: [1, 512]))
    let _ = try! model.prediction(from: input) // TODO: Check and see if the output is reasonable.
    total += (CFAbsoluteTimeGetCurrent() - start)
}

let average = total / Double(loopCount)
print("ComputUnit: \(unitName)")
print("Total: \(Measurement<UnitDuration>(value: total, unit: .seconds).formatted())")
print("Avg  : \(Measurement<UnitDuration>(value: average, unit: .seconds).converted(to: .milliseconds).formatted())")
