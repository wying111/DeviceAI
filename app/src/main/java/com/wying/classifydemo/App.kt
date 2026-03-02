package com.wying.classifydemo

import android.app.Application

class App: Application() {
    override fun onCreate() {
        super.onCreate()
        copyAssetToFile(this, "mobilenet_v2_trace_only.ptl")
    }
}