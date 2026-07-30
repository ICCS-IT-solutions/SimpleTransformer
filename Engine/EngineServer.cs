namespace SimpleTransformer.Engine
{
    //Todos for the Model subproject: 
    //Implement check point loading and resuming training from this point - priority objective.
    //Flesh out inference - once training can save and load. Inference needs to be able to load a checkpoint in order to work properly.
    //Bring in evaluation metrics for training and inference. This should enable better finetuning capabilities for the model.
    //Last: Create a console solution file for it so that I can AOT compile it without breaking the API which is managed code.
    //API subproject:
    //Update the API to use the new named pipe engine rather than the model directly.
    //Future objective: interprocess communication using network-based named pipes?
    public class EngineServer
    {
        //Read the user's request
        public void ReadRequest()
        {
            
        }
        //Execute the user's request against the model.
        public void Execute()
        {
            
        }
        public void WriteResponse()
        {
            
        }
    }
}