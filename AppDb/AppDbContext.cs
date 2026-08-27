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
        public DbSet<VocabularyEntry> VocabularyEntries { get; set; } = null!;
        public DbSet<TrainingConfigPresetEntry> TrainingConfigPresets { get; set; } = null!;
        public DbSet<TransformerConfigPresetEntry> TransformerConfigPresets { get; set; } = null!;
        public DbSet<TrainingJobEntry> TrainingJobs { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            //Training config
            modelBuilder.Entity<TrainingConfigEntry>()
            .Property(p => p.Config)
            .HasConversion<JsonStringValueConverter<TrainingConfig>>();
            //Transformer config
            modelBuilder.Entity<TransformerConfigEntry>()
            .Property(p => p.Config)
            .HasConversion<JsonStringValueConverter<TransformerConfig>>();
            //Presets - should be unique
            modelBuilder.Entity<TrainingConfigPresetEntry>()
            .HasIndex(i => i.Name)
            .IsUnique();
            modelBuilder.Entity<TransformerConfigPresetEntry>()
            .HasIndex(i => i.Name)
            .IsUnique();            
        }
    }
}