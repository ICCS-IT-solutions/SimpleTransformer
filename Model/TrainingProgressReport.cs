public class TrainingProgressReport
        {
            public int CurrentEpoch { get; }
            public int TotalEpochs { get; }
            public float Loss { get; }
            public TimeSpan ElapsedTime { get; }

            public TrainingProgressReport(int CurrentEpoch, int TotalEpochs, float Loss, TimeSpan ElapsedTime)
            {
                this.CurrentEpoch = CurrentEpoch;
                this.TotalEpochs = TotalEpochs;
                this.Loss = Loss;
                this.ElapsedTime = ElapsedTime;
            }
        }