package com.wying.classifydemo

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale

class MainActivity : ComponentActivity() {

    private var modelPointer: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val modelPath = filesDir.absolutePath + "/mobilenet_v2_trace_only.ptl"
        // 初始化模型，拿到 C++ 层的指针地址
        modelPointer = NativePredictor.loadModel(modelPath)

        setContent {

            val testBitmap = remember { createDummyBitmap(this@MainActivity) }

            var resultText by remember { mutableStateOf("等待推理...") }

            Column(modifier = Modifier.padding(16.dp)) {
                Image(bitmap = testBitmap.asImageBitmap(), contentDescription = "Test Image")

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = {
                    val classIndex = NativePredictor.predictImage(modelPointer, testBitmap)
                    resultText = "预测类别索引: $classIndex"
                }) {
                    Text("开始底层 C++ 推理")
                }

                Text(text = resultText, modifier = Modifier.padding(top = 16.dp))

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = {
                    val intent = Intent(this@MainActivity, CameraActivity::class.java)
                    startActivity(intent)
                }) {
                    Text("📷 打开相机扫描")
                }
            }
        }
    }


    fun createDummyBitmap(context: Context): Bitmap {
        val options = BitmapFactory.Options()
        options.inPreferredConfig = Bitmap.Config.ARGB_8888
        options.inScaled = false
        val rawBitmap = BitmapFactory.decodeResource(context.resources, R.mipmap.fish, options)
        return rawBitmap.scale(224, 224)
    }
}
