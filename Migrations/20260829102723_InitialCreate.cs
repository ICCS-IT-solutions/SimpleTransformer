using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "TrainingCheckpoints",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Filename = table.Column<string>(type: "TEXT", nullable: false),
                    Filepath = table.Column<string>(type: "TEXT", nullable: false),
                    Sha256 = table.Column<string>(type: "TEXT", nullable: true),
                    FileSize = table.Column<long>(type: "INTEGER", nullable: false),
                    Epoch = table.Column<int>(type: "INTEGER", nullable: false),
                    Loss = table.Column<float>(type: "REAL", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false),
                    TrainingRunId = table.Column<Guid>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TrainingCheckpoints", x => x.EntryId);
                });

            migrationBuilder.CreateTable(
                name: "TrainingConfigPresets",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    TrainingConfigEntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TrainingConfigPresets", x => x.EntryId);
                });

            migrationBuilder.CreateTable(
                name: "TrainingConfigs",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Description = table.Column<string>(type: "TEXT", nullable: false),
                    Config = table.Column<string>(type: "TEXT", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TrainingConfigs", x => x.EntryId);
                });

            migrationBuilder.CreateTable(
                name: "TransformerConfigPresets",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    TransformerConfigId = table.Column<Guid>(type: "TEXT", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TransformerConfigPresets", x => x.EntryId);
                });

            migrationBuilder.CreateTable(
                name: "TransformerConfigs",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Description = table.Column<string>(type: "TEXT", nullable: false),
                    Config = table.Column<string>(type: "TEXT", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TransformerConfigs", x => x.EntryId);
                });

            migrationBuilder.CreateTable(
                name: "Vocabularies",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    TokenizerType = table.Column<int>(type: "INTEGER", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false),
                    NumTokens = table.Column<int>(type: "INTEGER", nullable: false),
                    Filename = table.Column<string>(type: "TEXT", nullable: false),
                    Filepath = table.Column<string>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Vocabularies", x => x.EntryId);
                });

            migrationBuilder.CreateTable(
                name: "TrainingJobs",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    TransformerConfigId = table.Column<Guid>(type: "TEXT", nullable: false),
                    TrainingConfigId = table.Column<Guid>(type: "TEXT", nullable: false),
                    VocabularyId = table.Column<Guid>(type: "TEXT", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false),
                    DateStarted = table.Column<DateTime>(type: "TEXT", nullable: true),
                    DateCompleted = table.Column<DateTime>(type: "TEXT", nullable: true),
                    Status = table.Column<int>(type: "INTEGER", nullable: false),
                    CurrentEpoch = table.Column<int>(type: "INTEGER", nullable: false),
                    EpochsCompleted = table.Column<int>(type: "INTEGER", nullable: false),
                    TotalEpochs = table.Column<int>(type: "INTEGER", nullable: false),
                    CurrentBatch = table.Column<int>(type: "INTEGER", nullable: false),
                    BatchesCompleted = table.Column<int>(type: "INTEGER", nullable: false),
                    TotalBatches = table.Column<int>(type: "INTEGER", nullable: false),
                    CurrentSubBatch = table.Column<int>(type: "INTEGER", nullable: false),
                    SubBatchesCompleted = table.Column<int>(type: "INTEGER", nullable: false),
                    TotalSubBatches = table.Column<int>(type: "INTEGER", nullable: false),
                    CurrentLoss = table.Column<float>(type: "REAL", nullable: false),
                    TrainingCheckpointId = table.Column<Guid>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TrainingJobs", x => x.EntryId);
                    table.ForeignKey(
                        name: "FK_TrainingJobs_TrainingCheckpoints_TrainingCheckpointId",
                        column: x => x.TrainingCheckpointId,
                        principalTable: "TrainingCheckpoints",
                        principalColumn: "EntryId",
                        onDelete: ReferentialAction.SetNull);
                    table.ForeignKey(
                        name: "FK_TrainingJobs_TrainingConfigs_TrainingConfigId",
                        column: x => x.TrainingConfigId,
                        principalTable: "TrainingConfigs",
                        principalColumn: "EntryId",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_TrainingJobs_TransformerConfigs_TransformerConfigId",
                        column: x => x.TransformerConfigId,
                        principalTable: "TransformerConfigs",
                        principalColumn: "EntryId",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_TrainingJobs_Vocabularies_VocabularyId",
                        column: x => x.VocabularyId,
                        principalTable: "Vocabularies",
                        principalColumn: "EntryId",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateIndex(
                name: "IX_TrainingConfigPresets_Name",
                table: "TrainingConfigPresets",
                column: "Name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_Name",
                table: "TrainingJobs",
                column: "Name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_TrainingCheckpointId",
                table: "TrainingJobs",
                column: "TrainingCheckpointId");

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_TrainingConfigId",
                table: "TrainingJobs",
                column: "TrainingConfigId");

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_TransformerConfigId",
                table: "TrainingJobs",
                column: "TransformerConfigId");

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_VocabularyId",
                table: "TrainingJobs",
                column: "VocabularyId");

            migrationBuilder.CreateIndex(
                name: "IX_TransformerConfigPresets_Name",
                table: "TransformerConfigPresets",
                column: "Name",
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "TrainingConfigPresets");

            migrationBuilder.DropTable(
                name: "TrainingJobs");

            migrationBuilder.DropTable(
                name: "TransformerConfigPresets");

            migrationBuilder.DropTable(
                name: "TrainingCheckpoints");

            migrationBuilder.DropTable(
                name: "TrainingConfigs");

            migrationBuilder.DropTable(
                name: "TransformerConfigs");

            migrationBuilder.DropTable(
                name: "Vocabularies");
        }
    }
}
