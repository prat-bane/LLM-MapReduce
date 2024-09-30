package mapreduce

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat

import java.io.File

object CosineSimilarityDriver {
  def main(args: Array[String]): Unit = {
   /* if (args.length != 3) {
      System.err.println("Usage: CosineSimilarityDriver <input path> <output path> <cache file path>")
      System.exit(-1)
    }
*/
    val conf = new Configuration()
    val job = Job.getInstance(conf, "Cosine Similarity Computation")
    job.setJarByClass(CosineSimilarityDriver.getClass)

    // Set Mapper and Reducer classes
    job.setMapperClass(classOf[CosineSimilarityMapper])
    job.setReducerClass(classOf[CosineSimilarityReducer])

    // Set output key and value types
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    // Set input and output paths using FileInputFormat and FileOutputFormat
    FileInputFormat.addInputPath(job, new Path("D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\embeddings\\output"))
    FileOutputFormat.setOutputPath(job, new Path("D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\embeddings\\similarity\\output"))
   val relativePath = "D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\test.txt"

   // Create a File object
   val file = new File(relativePath)

   // Get the absolute path
   val absolutePath = file.getAbsolutePath

   println(s"Absolute Path: $absolutePath")

   // Get the absolute URI
   val absoluteURI = file.toURI

   println(s"Absolute URI: ${absoluteURI.toString}")
    // Add the all_embeddings.txt file to the Distributed Cache
    job.addCacheFile(absoluteURI)

    // Set number of reducers (adjust based on dataset size and cluster resources)
    job.setNumReduceTasks(4) // Example: using 4 reducers for scalability

    // Exit after completion
    System.exit(if (job.waitForCompletion(true)) 0 else 1)
  }
}
