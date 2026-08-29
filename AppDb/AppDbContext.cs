using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Model;

namespace SimpleTransformer.AppDb
{
    //For better future-proofing and being able to stop and resume training, along with register multiple jobs, I am going to create a database.
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
            
        }
        public DbSet<TrainingConfigEntry> TrainingConfigs { get; set; } = null!;
        public DbSet<TransformerConfigEntry> TransformerConfigs { get; set; } = null!;
        public DbSet<TrainingCheckpointEntry> TrainingCheckpoints { get; set; } = null!;
        public DbSet<VocabularyEntry> Vocabularies { get; set; } = null!;
        public DbSet<TrainingConfigPresetEntry> TrainingConfigPresets { get; set; } = null!;
        public DbSet<TransformerConfigPresetEntry> TransformerConfigPresets { get; set; } = null!;
        public DbSet<TrainingJobEntry> TrainingJobs { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // ---------------------------------------------------------------------
            // Training configuration
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingConfigEntry>()
                .Property(x => x.Config)
                .HasConversion<JsonStringValueConverter<TrainingConfig>>();

            // ---------------------------------------------------------------------
            // Transformer configuration
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TransformerConfigEntry>()
                .Property(x => x.Config)
                .HasConversion<JsonStringValueConverter<TransformerConfig>>();

            // ---------------------------------------------------------------------
            // Configuration presets
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingConfigPresetEntry>()
                .HasIndex(x => x.Name)
                .IsUnique();

            modelBuilder.Entity<TransformerConfigPresetEntry>()
                .HasIndex(x => x.Name)
                .IsUnique();

            // ---------------------------------------------------------------------
            // Training jobs
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.Name)
                .IsUnique();

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.TransformerConfigId);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.TrainingConfigId);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.VocabularyId);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.TrainingCheckpointId);

            // ---------------------------------------------------------------------
            // Relationships
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.TransformerConfig)
                .WithMany()
                .HasForeignKey(x => x.TransformerConfigId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.TrainingConfig)
                .WithMany()
                .HasForeignKey(x => x.TrainingConfigId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.Vocabulary)
                .WithMany()
                .HasForeignKey(x => x.VocabularyId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.TrainingCheckpoint)
                .WithMany()
                .HasForeignKey(x => x.TrainingCheckpointId)
                .OnDelete(DeleteBehavior.SetNull);
        }
    }
}