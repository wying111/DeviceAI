package com.wying.classifydemo

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.text.DecimalFormat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : ComponentActivity() {

    private var modelPointer: Long = 0L
    private lateinit var cameraExecutor: ExecutorService
    private var classNameMap: Map<Int, String> = emptyMap()

    // UI 状态
    private var resultText by mutableStateOf("正在启动相机...")
    private var lastAnalysisTime = 0L

    // 权限请求
    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                Toast.makeText(this, "相机权限已授予，开始扫描", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "需要相机权限才能使用扫描功能", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 初始化模型
        val modelPath = filesDir.absolutePath + "/mobilenet_v2_trace_only.ptl"
        modelPointer = NativePredictor.loadModel(modelPath)
        if (modelPointer == 0L) {
            Toast.makeText(this, "模型加载失败", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        Toast.makeText(this, "模型加载成功，准备扫描", Toast.LENGTH_SHORT).show()

        // 加载 imagenet 类别映射
        loadImageNetClasses()

        // 初始化线程池
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 请求相机权限
        cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)

        setContent {
            MaterialTheme {
                CameraScreen()
            }
        }
    }

    @Composable
    private fun CameraScreen() {
        Column(modifier = Modifier.fillMaxSize()) {
            // 相机预览区域
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                AndroidView(
                    factory = { context ->
                        PreviewView(context).also { previewView ->
                            startCamera(previewView)
                        }
                    },
                    modifier = Modifier.fillMaxSize()
                )
            }

            // 结果显示区域
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.surfaceVariant,
                tonalElevation = 4.dp
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "🔍 识别结果",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(MaterialTheme.colorScheme.primaryContainer)
                            .padding(16.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = resultText,
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            textAlign = TextAlign.Center,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "每 1 秒自动识别一次",
                        fontSize = 12.sp,
                        color = Color.Gray
                    )
                }
            }
        }
    }

    private fun startCamera(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 预览用例
            val preview = Preview.Builder()
                .build()
                .also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            // 图像分析用例
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        analyzeImage(imageProxy)
                    }
                }

            // 使用后置摄像头
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
                resultText = "相机已就绪，等待识别..."
            } catch (e: Exception) {
                resultText = "相机启动失败: ${e.message}"
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeImage(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // 每 3 秒进行一次识别
        if (currentTime - lastAnalysisTime < 1000) {
            imageProxy.close()
            return
        }
        lastAnalysisTime = currentTime

        try {
            // 将 ImageProxy 转换为 Bitmap
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                // 缩放到模型输入尺寸 224x224
                val scaledBitmap = bitmap.scale(224, 224)

                // 执行推理
                val resultMap = NativePredictor.predictImage(modelPointer, scaledBitmap)

                // 获取中文名称
                val className = buildString {
                    resultMap.toSortedMap().forEach {
                        append("${classNameMap[it.key]}:${it.value.toDouble().toPercentString()}\n")
                    }
                }

                // 在主线程更新 UI 和显示 Toast
                runOnUiThread {
                    resultText = className
//                    Toast.makeText(
//                        this@CameraActivity,
//                        "识别到: $className",
//                        Toast.LENGTH_SHORT
//                    ).show()
                }

                android.util.Log.d("CameraActivity", "识别结果: resultMap=$resultMap, name=$className")

                // 回收临时 bitmap
                if (scaledBitmap !== bitmap) {
                    scaledBitmap.recycle()
                }
                bitmap.recycle()
            }
        } catch (e: Exception) {
            android.util.Log.e("CameraActivity", "识别出错: ${e.message}", e)
            runOnUiThread {
                resultText = "识别出错: ${e.message}"
            }
        } finally {
            imageProxy.close()
        }
    }

    /**
     * 将 ImageProxy (YUV_420_888) 转换为 ARGB_8888 Bitmap
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(
            nv21,
            android.graphics.ImageFormat.NV21,
            imageProxy.width,
            imageProxy.height,
            null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height),
            100,
            out
        )

        val jpegBytes = out.toByteArray()
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)

        // 处理旋转
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        return if (rotationDegrees != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            bitmap.recycle()
            rotated
        } else {
            bitmap
        }
    }

    /**
     * 从 assets 加载 imagenet_classes.json 并构建 id -> 中文名称 的映射
     */
    private fun loadImageNetClasses() {
        try {
            val jsonString = assets.open("imagenet_classes.json").bufferedReader().use { it.readText() }
            val type = object : TypeToken<List<ImageNetClass>>() {}.type
            val classList: List<ImageNetClass> = Gson().fromJson(jsonString, type)
            classNameMap = classList.associate { it.id to it.name_cn }
            android.util.Log.d("CameraActivity", "加载了 ${classNameMap.size} 个 ImageNet 类别")
        } catch (e: Exception) {
            android.util.Log.e("CameraActivity", "加载类别文件失败: ${e.message}", e)
            Toast.makeText(this, "加载类别文件失败", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }
    }

    /**
     * imagenet_classes.json 的数据类
     */
    data class ImageNetClass(
        val id: Int,
        val name_en: String,
        val name_cn: String
    )
}

fun Double.toPercentString(): String {
    val df = DecimalFormat("#.##")
    return df.format(this * 100) + "%"
}