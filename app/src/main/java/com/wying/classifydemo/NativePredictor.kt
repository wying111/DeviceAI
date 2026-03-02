package com.wying.classifydemo

import android.graphics.Bitmap

object NativePredictor {
    // 加载我们自己编写的 C++ 动态链接库
    init {
        System.loadLibrary("ai_infra_core")
    }

    /**
     * 【Infra 视角：对象生命周期管理】
     * 为什么返回值是 Long？
     * 在 C++ 中加载模型会产生一个庞大的 Module 对象。我们不能每次推理都重新加载。
     * 因此，我们在 C++ 层 new 一个对象，把它的内存地址（指针）强转为 Long 返回给 Java 侧保存。
     * 每次推理时，再把这个内存地址传回给 C++，实现底层模型实例的复用。
     */
    external fun loadModel(modelPath: String): Long

    /**
     * 传入模型指针和 Android Bitmap，返回最高概率的类别索引 (Int)
     */
    external fun predictImage(modelPtr: Long, bitmap: Bitmap): Map<Int, Float>
}