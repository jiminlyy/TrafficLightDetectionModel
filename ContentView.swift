import SwiftUI
import AVFoundation
import Vision
import CoreML
import Combine

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var detector = TrafficLightDetector()

    var body: some View {
        ZStack {
            // 카메라 프리뷰
            CameraPreview(session: detector.captureSession)
                .ignoresSafeArea()

            // 바운딩 박스 오버레이
            GeometryReader { geo in
                ForEach(detector.detections) { det in
                    let rect = VNImageRectForNormalizedRect(
                        det.boundingBox,
                        Int(geo.size.width),
                        Int(geo.size.height)
                    )
                    // VNImageRectForNormalizedRect는 하단 기준이라 Y 뒤집기 필요
                    let flipped = CGRect(
                        x: rect.minX,
                        y: geo.size.height - rect.maxY,
                        width: rect.width,
                        height: rect.height
                    )

                    ZStack(alignment: .topLeading) {
                        Rectangle()
                            .stroke(det.color, lineWidth: 3)
                            .frame(width: flipped.width, height: flipped.height)
                            .position(x: flipped.midX, y: flipped.midY)

                        Text("\(det.label) \(Int(det.confidence * 100))%")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                            .padding(4)
                            .background(det.color)
                            .cornerRadius(4)
                            .position(x: flipped.minX + 40, y: flipped.minY - 10)
                    }
                }
            }
            .ignoresSafeArea()

            // 하단 상태 패널
            VStack {
                Spacer()
                VStack(spacing: 12) {
                    Circle()
                        .fill(detector.signalColor)
                        .frame(width: 60, height: 60)
                        .overlay(Circle().stroke(Color.white, lineWidth: 3))
                        .shadow(radius: 8)

                    Text(detector.statusText)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)

                    if detector.confidence > 0 {
                        Text("신뢰도: \(Int(detector.confidence * 100))%")
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                .padding(20)
                .background(RoundedRectangle(cornerRadius: 20).fill(Color.black.opacity(0.6)))
                .padding(.bottom, 50)
            }
        }
        .onAppear { detector.startDetection() }
        .onDisappear { detector.stopDetection() }
    }
}

// MARK: - Detection 모델
struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect  // Vision 정규화 좌표 (0~1)
    var color: Color {
        switch label {
        case "ped_green": return .green
        case "ped_red":   return .red
        default:          return .yellow
        }
    }
}

// MARK: - TrafficLightDetector
class TrafficLightDetector: NSObject, ObservableObject {
    @Published var statusText: String = "신호등을 찾는 중..."
    @Published var signalColor: Color = .gray
    @Published var confidence: Float = 0
    @Published var detections: [Detection] = []

    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "video.processing", qos: .userInitiated)

    private var visionModel: VNCoreMLModel?
    private let synthesizer = AVSpeechSynthesizer()

    private var lastSpokenSignal: String = ""
    private var lastSpeakTime: Date = .distantPast
    private let speakInterval: TimeInterval = 3.0

    private var lastDetectionTime: Date = Date()
    private let noDetectionTimeout: TimeInterval = 5.0

    override init() {
        super.init()
        setupModel()
        setupCamera()
        setupAudio()
    }

    private func setupModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let mlModel = try best(configuration: config).model
            visionModel = try VNCoreMLModel(for: mlModel)
        } catch {
            print("모델 로드 실패: \(error)")
        }
    }
 
    private func setupAudio() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: .mixWithOthers)
            try session.setActive(true)
        } catch {
            print("오디오 세션 설정 실패: \(error)")
        }
    }

    private func setupCamera() {
        captureSession.sessionPreset = .hd1280x720

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("카메라 접근 불가")
            return
        }

        if captureSession.canAddInput(input) { captureSession.addInput(input) }

        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90
        }
    }

    func startDetection() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    func stopDetection() {
        captureSession.stopRunning()
    }

    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let model = visionModel else { return }

        let request = VNCoreMLRequest(model: model) { [weak self] request, _ in
            self?.handleResults(request.results)
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }

    private func handleResults(_ results: [VNObservation]?) {
        guard let observations = results as? [VNRecognizedObjectObservation] else { return }

        let filtered = observations.compactMap { obs -> Detection? in
            guard let label = obs.labels.first, label.confidence >= 0.5 else { return nil }
            return Detection(
                label: label.identifier,
                confidence: label.confidence,
                boundingBox: obs.boundingBox
            )
        }

        let best = filtered.max(by: { $0.confidence < $1.confidence })

        DispatchQueue.main.async {
            self.detections = filtered  // 바운딩 박스 전체 업데이트

            if let det = best {
                self.lastDetectionTime = Date()
                self.confidence = det.confidence

                switch det.label {
                case "ped_green":
                    self.statusText = "초록불 — 건너세요"
                    self.signalColor = .green
                    self.speak("초록불입니다. 건너세요.", signal: "green")
                case "ped_red":
                    self.statusText = "빨간불 — 기다리세요"
                    self.signalColor = .red
                    self.speak("빨간불입니다. 기다리세요.", signal: "red")
                default:
                    self.statusText = "신호등을 찾는 중..."
                    self.signalColor = .gray
                    self.confidence = 0
                }
            } else {
                if Date().timeIntervalSince(self.lastDetectionTime) > self.noDetectionTimeout {
                    self.statusText = "신호등을 찾는 중..."
                    self.signalColor = .gray
                    self.confidence = 0
                    self.lastSpokenSignal = ""
                    self.detections = []
                }
            }
        }
    }

    private func speak(_ text: String, signal: String) {
        let now = Date()
        if signal == lastSpokenSignal && now.timeIntervalSince(lastSpeakTime) < speakInterval { return }
        if signal != lastSpokenSignal && synthesizer.isSpeaking { synthesizer.stopSpeaking(at: .immediate) }
        lastSpokenSignal = signal
        lastSpeakTime = now
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "ko-KR")
        utterance.rate = 0.5
        utterance.pitchMultiplier = 1.1
        synthesizer.speak(utterance)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension TrafficLightDetector: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer)
    }
}

// MARK: - CameraPreview
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

// AVCaptureVideoPreviewLayer를 직접 레이어로 쓰는 UIView
// → bounds 변화에 자동 반응해서 카메라 화면이 꽉 차게 표시됨
class PreviewUIView: UIView {
    var session: AVCaptureSession? {
        didSet {
            guard let session else { return }
            previewLayer.session = session
        }
    }

    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = bounds
    }
}

#Preview {
    ContentView()
}
