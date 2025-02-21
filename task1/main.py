from training.trainer import ModelTrainer

def main():
    trainer = ModelTrainer()
    
    try:
        algorithm = trainer.get_algorithm_choice()
        print(f"\nSelected algorithm: {algorithm}")
        
        results = trainer.train_and_evaluate(algorithm)
        trainer.print_results(results)
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()