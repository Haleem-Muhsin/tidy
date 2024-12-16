/**
 * Copyright 2023 Viacheslav Barkov
 */

package com.slavabarkov.tidy.viewmodels

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.content.ContentResolver
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import androidx.lifecycle.*
import com.slavabarkov.tidy.R
import com.slavabarkov.tidy.centerCrop
import com.slavabarkov.tidy.data.ImageEmbedding
import com.slavabarkov.tidy.data.ImageEmbeddingDatabase
import com.slavabarkov.tidy.data.ImageEmbeddingRepository
import com.slavabarkov.tidy.normalizeL2
import com.slavabarkov.tidy.preProcess
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.*

class ORTImageViewModel(application: Application) : AndroidViewModel(application) {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var repository: ImageEmbeddingRepository
    var idxList: ArrayList<Long> = arrayListOf()
    var embeddingsList: ArrayList<FloatArray> = arrayListOf()
    var progress: MutableLiveData<Double> = MutableLiveData(0.0)

    init {
        val imageEmbeddingDao = ImageEmbeddingDatabase.getDatabase(application).imageEmbeddingDao()
        repository = ImageEmbeddingRepository(imageEmbeddingDao)
    }

    fun generateIndex() {
        val modelID = R.raw.visual_quant
        val resources = getApplication<Application>().resources
        val model = resources.openRawResource(modelID).readBytes()
        val session = ortEnv.createSession(model)

        viewModelScope.launch(Dispatchers.Main) {
            progress.value = 0.0
            val uri: Uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
            val projection = arrayOf(
                MediaStore.Images.Media._ID,
                MediaStore.Images.Media.DATE_MODIFIED,
                MediaStore.Images.Media.BUCKET_DISPLAY_NAME
            )
            val sortOrder = "${MediaStore.Images.Media._ID} ASC"
            val contentResolver: ContentResolver = getApplication<Application>().contentResolver
            val cursor: Cursor? = contentResolver.query(uri, projection, null, null, sortOrder)
            val totalImages = cursor?.count ?: 0
            cursor?.use {
                val idColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                val dateColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)
                val bucketColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME)

                while (it.moveToNext()) {
                    val id: Long = it.getLong(idColumn)
                    val date: Long = it.getLong(dateColumn)
                    // Safely get bucket name, defaulting to empty string if null
                    val bucket: String = if (it.isNull(bucketColumn)) "" else it.getString(bucketColumn) ?: ""

                    // Skip screenshots folder if bucket name matches
                    if (bucket.equals("Screenshots", ignoreCase = true)) continue

                    val record = repository.getRecord(id) as ImageEmbedding?
                    if (record != null) {
                        idxList.add(record.id)
                        embeddingsList.add(record.embedding)
                    } else {
                        val imageUri: Uri = Uri.withAppendedPath(uri, id.toString())
                        try {
                            contentResolver.openInputStream(imageUri)?.use { inputStream ->
                                val bytes = inputStream.readBytes()

                                // Can fail to create the image decoder if its not implemented for the image type
                                val bitmap: Bitmap? = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                                bitmap?.let {
                                    val rawBitmap = centerCrop(bitmap, 224)
                                    val inputShape = longArrayOf(1, 3, 224, 224)
                                    val inputName = "pixel_values"
                                    val imgData = preProcess(rawBitmap)

                                    OnnxTensor.createTensor(ortEnv, imgData, inputShape).use { inputTensor ->
                                        val output = session?.run(Collections.singletonMap(inputName, inputTensor))
                                        output?.use {
                                            @Suppress("UNCHECKED_CAST")
                                            var rawOutput = ((output[0]?.value) as Array<FloatArray>)[0]
                                            rawOutput = normalizeL2(rawOutput)
                                            repository.addImageEmbedding(
                                                ImageEmbedding(
                                                    id, date, rawOutput
                                                )
                                            )
                                            idxList.add(id)
                                            embeddingsList.add(rawOutput)
                                        }
                                    }
                                }
                                bitmap?.recycle()
                            }
                        } catch (e: Exception) {
                            // Log error or handle exception as needed
                            e.printStackTrace()
                            continue
                        }
                    }
                    // Record created/loaded, update progress
                    progress.value = it.position.toDouble() / totalImages.toDouble()
                }
            }
            cursor?.close()
            session.close()
            progress.value = 1.0
        }
    }
}

