package mapreduce

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, LocatedFileStatus, Path, RemoteIterator}
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.slf4j.{LoggerFactory}

import java.net.URI

object CosineSimilarityDriver {
  private val logger = LoggerFactory.getLogger(CosineSimilarityDriver.getClass)
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      logger.error("Invalid arguments. Usage: CosineSimilarityDriver <input path> <output path> <cache file path>")
      System.exit(-1)
    }


    val conf = new Configuration()
    val job = Job.getInstance(conf, "Cosine Similarity Computation")
    job.setJarByClass(CosineSimilarityDriver.getClass)
    logger.info("Job instance created for Cosine Similarity Computation.")


    // Set Mapper and Reducer classes
    job.setMapperClass(classOf[CosineSimilarityMapper])
    job.setReducerClass(classOf[CosineSimilarityReducer])

    // Set output key and value types
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    // Set input and output paths using FileInputFormat and FileOutputFormat
    FileInputFormat.addInputPath(job, new Path(args(0)))
    FileOutputFormat.setOutputPath(job, new Path(args(1)))

    //val path = findFilePath("hdfs://localhost:9000","/input/cosine","embeddings.txt")



    // Adding the all_embeddings.txt file to the Distributed Cache
    job.addCacheFile(new Path(args(2)).toUri)
    logger.info(s"Cache file has been added to the Distributed Cache.")

    // Set number of reducers (adjust based on dataset size and cluster resources)
    job.setNumReduceTasks(4) // Example: using 4 reducers for scalability

    // Exit after completion
    System.exit(if (job.waitForCompletion(true)) 0 else 1)
  }

}
